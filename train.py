# -*- coding: utf-8 -*-
"""
字符级 Transformer 语言模型训练。
依赖：同目录下已先生成 structured_*.pt / structured_vocab.json；model.py、dataset.py。
"""
import argparse
import os
import random
import sys
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PoetryBlockDataset, load_vocab_json
from model import CharGPT
from poetry_format import build_loss_weight_by_token_id
from prosody import NUM_RHYMES, NUM_TONES

# =============================================================================
# 默认可调超参：优先在此修改。下方 argparse 的 default 会引用这些常量。
# 命令行仍可覆盖，例如：python train.py --lr 1e-3 --epochs 5
# =============================================================================

# ---- 数据与文件路径 ----
# prepare_data.py 产出的单条长 id 序列张量
TRAIN_PT = "structured_train_data.pt"
VAL_PT = "structured_val_data.pt"
TRAIN_TONE_PT = "structured_train_tone.pt"
VAL_TONE_PT = "structured_val_tone.pt"
TRAIN_RHYME_PT = "structured_train_rhyme.pt"
VAL_RHYME_PT = "structured_val_rhyme.pt"
# 字符 <-> id 映射与词表大小
VOCAB = "structured_vocab.json"
# 验证 loss 更优时写入的 PyTorch 权重，predict.py 默认加载同名文件
CKPT_OUT = "structured_ckpt_best.pt"
# 每轮 epoch 的 train/val loss 折线图
LOSS_PLOT = "structured_loss_curve.png"

# ---- 模型结构（与 CharGPT、prepare 时的 block 一致，改后需能重新训练/兼容 checkpoint） ----
# 每段输入/输出的字符长度（时间步 T）
BLOCK_SIZE = 128
# 词嵌入与注意力隐藏维度 d_model
D_MODEL = 256
# 注意力头数；须满足 D_MODEL % N_HEAD == 0
N_HEAD = 8
# Transformer 层堆叠数
N_LAYER = 4
# 前馈子层上投影维度（FFN 中间层宽度）
D_FF = 1024
# 注意力/FFN 中 Dropout 比例，0~1
DROPOUT = 0.1

# ---- 训练过程 ----
# 每步样本条数，越大占显存/内存越多
BATCH_SIZE = 32
# AdamW 学习率
LR = 3e-4
# 完整遍历“当前训练集长度”的次数（见下 TRAIN_NUM_SAMPLES 与 MAX_TRAIN_BATCHES）
EPOCHS = 3
# 随机数种子，调参对比时可固定以便复现
SEED = 42

# 每轮在验证集上只跑前几个 batch 估计 val_loss，加速调参；0=整份验证集前向，通常很慢
VAL_BATCHES = 200

# 将长语料切成的训练「窗口」条数；在序列上随机起窗。远小于“全部滑窗”时可让单 epoch 可接受
# 0=不限制、长度约为「字符数 - BLOCK_SIZE」条，单 epoch 可能上百万 step
TRAIN_NUM_SAMPLES = 50_000

# 每轮从训练 DataLoader 里最多只跑前若干个 batch 就记 train_loss 并做验证，用于快速试跑
# 0=按 DataLoader 长度跑满本 epoch
MAX_TRAIN_BATCHES = 0

# 未安装 tqdm 时，无进度条，每隔多少 batch 打印一次当前步 loss；0=不打印
LOG_INTERVAL = 50

# ---- 优化与数值稳定（无对应命令行参数，仅改此处） ----
# L2 权重衰减，抑制过拟合
WEIGHT_DECAY = 0.01
# 全局梯度范数上限，超则按比例缩放，减轻爆炸
GRAD_CLIP_NORM = 1.0

# ======== Form-Stressed Weighting Config: START ========
# 变更说明:
# - 参考 JiuGeGPT，对结构 token 的 loss 施加额外权重。
# - 默认：行分隔符=2，阕分隔符=2，结束符=3。
USE_FORM_WEIGHTING = True
LINE_SEP_WEIGHT = 2.0
STANZA_SEP_WEIGHT = 2.0
EOS_WEIGHT = 3.0
# ======== Form-Stressed Weighting Config: END ========

# ======== Prosody Training Config: START ========
# 变更说明:
# - 新增 Tone / Rhyme embedding 与辅助损失开关。
# - 默认开启，生成阶段不做硬约束，只作为软条件输入。
USE_PROSODY = True
USE_AUX_LOSS = True
AUX_LOSS_WEIGHT = 0.1
# ======== Prosody Training Config: END ========

# ---- DataLoader 行为（无对应命令行参数） ----
# 0=主进程加载；>0 为子进程数，在 Windows/部分环境上可能需 0
DATALOADER_NUM_WORKERS = 0
# 若该轮样本数不是 BATCH_SIZE 整数倍，是否丢弃最后不足一批；True 使每步张量 batch 维一致
DATALOADER_DROP_LAST = True

# =============================================================================


def set_seed(s: int) -> None:
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ======== Prosody Batch Parser: START ========
# 变更说明:
# - 统一解析 dataset 返回的普通/加权/带韵律三种 batch 形态。
# - 训练与验证共用，避免 forward 参数分支散落在多个位置。
def unpack_batch(batch, device: torch.device):
    weights = None
    tone_x = tone_y = rhyme_x = rhyme_y = None
    if len(batch) == 7:
        x, y, weights, tone_x, tone_y, rhyme_x, rhyme_y = batch
    elif len(batch) == 6:
        x, y, tone_x, tone_y, rhyme_x, rhyme_y = batch
    elif len(batch) == 3:
        x, y, weights = batch
    else:
        x, y = batch
    x = x.to(device)
    y = y.to(device)
    if weights is not None:
        weights = weights.to(device)
    if tone_x is not None:
        tone_x = tone_x.to(device)
        tone_y = tone_y.to(device)
        rhyme_x = rhyme_x.to(device)
        rhyme_y = rhyme_y.to(device)
    return x, y, weights, tone_x, tone_y, rhyme_x, rhyme_y


def build_forward_kwargs(weights, tone_x, tone_y, rhyme_x, rhyme_y):
    kwargs = {}
    if weights is not None:
        kwargs["loss_weights"] = weights
    if tone_x is not None:
        kwargs["tone_ids"] = tone_x
        kwargs["rhyme_ids"] = rhyme_x
        kwargs["tone_targets"] = tone_y
        kwargs["rhyme_targets"] = rhyme_y
    return kwargs
# ======== Prosody Batch Parser: END ========


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x, y, w, tone_x, tone_y, rhyme_x, rhyme_y = unpack_batch(batch, device)
        _, loss = model(x, y, **build_forward_kwargs(w, tone_x, tone_y, rhyme_x, rhyme_y))
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(
        description="超参数默认来自文件顶部 CONFIG 区域，可用命令行覆盖见 --help。",
    )
    p.add_argument("--train_pt", type=str, default=TRAIN_PT)
    p.add_argument("--val_pt", type=str, default=VAL_PT)
    p.add_argument("--train_tone_pt", type=str, default=TRAIN_TONE_PT)
    p.add_argument("--val_tone_pt", type=str, default=VAL_TONE_PT)
    p.add_argument("--train_rhyme_pt", type=str, default=TRAIN_RHYME_PT)
    p.add_argument("--val_rhyme_pt", type=str, default=VAL_RHYME_PT)
    p.add_argument("--vocab", type=str, default=VOCAB)
    p.add_argument("--save", type=str, default=CKPT_OUT)
    p.add_argument("--block_size", type=int, default=BLOCK_SIZE)
    p.add_argument("--d_model", type=int, default=D_MODEL)
    p.add_argument("--n_head", type=int, default=N_HEAD)
    p.add_argument("--n_layer", type=int, default=N_LAYER)
    p.add_argument("--d_ff", type=int, default=D_FF)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument(
        "--val_batches",
        type=int,
        default=VAL_BATCHES,
        help="每轮验证集上最多前向多少个 batch；0 表示跑完全部验证集（可能很慢）",
    )
    p.add_argument(
        "--train_num_samples",
        type=int,
        default=TRAIN_NUM_SAMPLES,
        help="训练集每 epoch 使用的窗口条数；在长序列上随机起窗。0=使用全部滑窗（可上千万条，每 epoch 极慢）",
    )
    p.add_argument(
        "--max_train_batches",
        type=int,
        default=MAX_TRAIN_BATCHES,
        help="每轮训练最多走多少个 batch 后进入验证；0 表示走满当前 DataLoader 长度",
    )
    p.add_argument(
        "--log_interval",
        type=int,
        default=LOG_INTERVAL,
        help="无 tqdm 时每多少个 batch 打印一次训练 loss；0 表示不打印",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--log_plot", type=str, default=LOSS_PLOT, help="保存 loss 曲线图路径")
    p.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help="从已有 checkpoint 加载模型权重后继续训练；若 checkpoint 含 optimizer/epoch，也会一并恢复",
    )
    p.add_argument(
        "--disable_form_weighting",
        action="store_true",
        help="关闭 form-stressed weighting，退回普通交叉熵",
    )
    p.add_argument("--line_sep_weight", type=float, default=LINE_SEP_WEIGHT)
    p.add_argument("--stanza_sep_weight", type=float, default=STANZA_SEP_WEIGHT)
    p.add_argument("--eos_weight", type=float, default=EOS_WEIGHT)
    p.add_argument(
        "--disable_prosody",
        action="store_true",
        help="关闭 Tone/Rhyme embedding，退回纯字符模型",
    )
    p.add_argument(
        "--disable_aux_loss",
        action="store_true",
        help="关闭 Tone/Rhyme 辅助损失，仅保留 prosody embedding",
    )
    p.add_argument("--aux_loss_weight", type=float, default=AUX_LOSS_WEIGHT)
    args = p.parse_args()

    required_files = [args.train_pt, args.val_pt, args.vocab]
    use_prosody = USE_PROSODY and (not args.disable_prosody)
    if use_prosody:
        required_files.extend(
            [args.train_tone_pt, args.val_tone_pt, args.train_rhyme_pt, args.val_rhyme_pt]
        )
    for f in required_files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"缺少 {f}，请先运行 build_structured_dataset.py")

    set_seed(args.seed)
    device = get_device()
    print("设备:", device)

    stoi, _, vs = load_vocab_json(args.vocab)
    vocab_size = vs
    use_form_weighting = USE_FORM_WEIGHTING and (not args.disable_form_weighting)
    use_aux_loss = USE_AUX_LOSS and use_prosody and (not args.disable_aux_loss)
    form_weight_map = (
        build_loss_weight_by_token_id(
            stoi,
            line_sep_weight=args.line_sep_weight,
            stanza_sep_weight=args.stanza_sep_weight,
            eos_weight=args.eos_weight,
        )
        if use_form_weighting
        else {}
    )
    if use_form_weighting:
        print(f"启用 form-stressed weighting: {form_weight_map}", flush=True)
    else:
        print("未启用 form-stressed weighting", flush=True)
    if use_prosody:
        print(
            f"启用 prosody embedding: tone={args.train_tone_pt}, rhyme={args.train_rhyme_pt}",
            flush=True,
        )
        print(
            f"启用 prosody auxiliary loss={use_aux_loss}, aux_loss_weight={args.aux_loss_weight:.3f}",
            flush=True,
        )
    else:
        print("未启用 prosody embedding / auxiliary loss", flush=True)

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore

    train_n = int(args.train_num_samples) if int(args.train_num_samples) > 0 else None
    print("正在加载训练/验证张量 (structured_train_data.pt 可能较大，需等待片刻)...", flush=True)
    train_ds = PoetryBlockDataset(
        args.train_pt,
        args.block_size,
        num_samples=train_n,
        sample_random=True,
        target_token_weights=form_weight_map,
        tone_path=args.train_tone_pt if use_prosody else None,
        rhyme_path=args.train_rhyme_pt if use_prosody else None,
    )
    val_ds = PoetryBlockDataset(
        args.val_pt,
        args.block_size,
        num_samples=None,
        sample_random=False,
        target_token_weights=form_weight_map,
        tone_path=args.val_tone_pt if use_prosody else None,
        rhyme_path=args.val_rhyme_pt if use_prosody else None,
    )
    ntok_tr = int(train_ds.data.size(0))  # type: ignore[attr-defined]
    nbatch_tr = (len(train_ds) + args.batch_size - 1) // args.batch_size
    nbatch_effective = nbatch_tr
    if int(args.max_train_batches) > 0:
        nbatch_effective = min(nbatch_effective, int(args.max_train_batches))
    print(
        f"训练序列长度(字符 token 数)≈{ntok_tr}\n"
        f"本轮 DataLoader: len(dset)={len(train_ds):,}，batch_size={args.batch_size}，"
        f"每 epoch 约 {nbatch_tr:,} 个 batch"
        + (f"（`--max_train_batches` 截断为 {nbatch_effective}）" if nbatch_effective < nbatch_tr else ""),
        flush=True,
    )
    if train_n is None:
        print(
            "注意: 你正在使用**全部**滑窗；单 epoch 可能达百万级 step，首个 Epoch 完成前"
            "终端可能长时间无新输出。建议改用 --train_num_samples 50000 等。",
            file=sys.stderr,
            flush=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=DATALOADER_NUM_WORKERS,
        drop_last=DATALOADER_DROP_LAST,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        drop_last=DATALOADER_DROP_LAST,
    )

    model = CharGPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_prosody=use_prosody,
        num_tones=NUM_TONES,
        num_rhymes=NUM_RHYMES + 1,
        use_aux_loss=use_aux_loss,
        aux_loss_weight=args.aux_loss_weight,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")
    start_epoch = 1

    # ======== Resume / Warm Start: START ========
    # 变更说明:
    # - 支持从已有 checkpoint 热启动继续训练。
    # - 若 checkpoint 含 optimizer/epoch/best_val，则尽量恢复；否则退化为仅加载模型权重。
    if args.init_ckpt:
        if not os.path.isfile(args.init_ckpt):
            raise FileNotFoundError(f"找不到 init_ckpt: {args.init_ckpt}")
        pack = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(pack["model"], strict=True)
        if "optimizer" in pack:
            opt.load_state_dict(pack["optimizer"])
        if isinstance(pack.get("epoch"), int):
            start_epoch = int(pack["epoch"]) + 1
        if pack.get("best_val") is not None:
            best_val = float(pack["best_val"])
        print(
            f"已加载初始 checkpoint: {args.init_ckpt} "
            f"(start_epoch={start_epoch}, best_val={best_val:.4f})",
            flush=True,
        )
    # ======== Resume / Warm Start: END ========

    end_epoch = start_epoch + args.epochs - 1
    for ep in range(start_epoch, start_epoch + args.epochs):
        model.train()
        run = 0.0
        cnt = 0
        max_tb = int(args.max_train_batches) if int(args.max_train_batches) > 0 else 0
        it = train_loader
        if tqdm is not None:
            it = tqdm(  # type: ignore[call-arg, misc]
                train_loader,
                desc=f"Epoch {ep}/{end_epoch}",
                total=max_tb or len(train_loader),
                file=sys.stdout,
            )
        bi = 0
        for batch in it:
            x, y, w, tone_x, tone_y, rhyme_x, rhyme_y = unpack_batch(batch, device)
            opt.zero_grad()
            _, loss = model(x, y, **build_forward_kwargs(w, tone_x, tone_y, rhyme_x, rhyme_y))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            run += loss.item() * x.size(0)
            cnt += x.size(0)
            if int(args.log_interval) > 0 and tqdm is None and (bi + 1) % int(args.log_interval) == 0:
                print(
                    f"  epoch {ep}  step {bi + 1}  loss {loss.item():.4f}",
                    flush=True,
                )
            bi += 1
            if max_tb and bi >= max_tb:
                break
        tr = run / max(cnt, 1)
        max_vb = args.val_batches if args.val_batches > 0 else 0
        vb = eval_loss(model, val_loader, device, max_batches=max_vb)
        train_losses.append(tr)
        val_losses.append(vb)
        print(f"Epoch {ep}/{end_epoch}  train_loss={tr:.4f}  val_loss={vb:.4f}")
        if vb < best_val:
            best_val = vb
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": ep,
                    "best_val": best_val,
                    "hparams": {
                        "vocab_size": vocab_size,
                        "block_size": args.block_size,
                        "d_model": args.d_model,
                        "n_head": args.n_head,
                        "n_layer": args.n_layer,
                        "d_ff": args.d_ff,
                        "dropout": args.dropout,
                        "use_form_weighting": use_form_weighting,
                        "use_prosody": use_prosody,
                        "num_tones": NUM_TONES,
                        "num_rhymes": NUM_RHYMES + 1,
                        "use_aux_loss": use_aux_loss,
                        "aux_loss_weight": args.aux_loss_weight,
                    },
                },
                args.save,
            )
            print(f"  -> 保存更优模型  val_loss={vb:.4f}  至 {args.save}")

    if train_losses and val_losses:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.log_plot, dpi=150)
        print("已保存曲线:", os.path.abspath(args.log_plot))
    print("训练结束。最佳 val_loss（本轮记录）: {:.4f}".format(min(val_losses) if val_losses else 0.0))


if __name__ == "__main__":
    main()
