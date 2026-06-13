# -*- coding: utf-8 -*-
"""
在已有结构化韵律模型上做主题 LoRA 微调。
默认策略：
1. 加载 `structured_ckpt_50e_prosody.pt` 作为底座；
2. 新增 theme embedding；
3. 对 Transformer 主干挂接 LoRA；
4. 冻结原始参数，只训练 `theme_emb + LoRA`。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PoetryBlockDataset, load_vocab_json
from model import CharGPT
from poetry_format import build_loss_weight_by_token_id
from prosody import NUM_RHYMES, NUM_TONES
from train import get_device, set_seed

TRAIN_PT = "theme_train_data.pt"
VAL_PT = "theme_val_data.pt"
TRAIN_TONE_PT = "theme_train_tone.pt"
VAL_TONE_PT = "theme_val_tone.pt"
TRAIN_RHYME_PT = "theme_train_rhyme.pt"
VAL_RHYME_PT = "theme_val_rhyme.pt"
TRAIN_THEME_PT = "theme_train_theme.pt"
VAL_THEME_PT = "theme_val_theme.pt"
THEME_VOCAB = "theme_vocab.json"
VOCAB = "structured_vocab.json"
BASE_CKPT = "structured_ckpt_50e_prosody.pt"
SAVE_CKPT = "theme_lora_ckpt_best.pt"
LOSS_PLOT = "theme_lora_loss_curve.png"

BLOCK_SIZE = 128
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 30
TRAIN_NUM_SAMPLES = 10000
VAL_BATCHES = 50
SEED = 42
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0

LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.05

LINE_SEP_WEIGHT = 2.0
STANZA_SEP_WEIGHT = 2.0
EOS_WEIGHT = 3.0


# ======== Theme LoRA Batch Parser: START ========
# 变更说明:
# - 主题微调在普通 token/prosody 之外，还会返回 theme_id 窗口。
# - 这里统一解包，避免训练/验证逻辑分叉。
def unpack_theme_batch(batch, device: torch.device):
    weights = None
    theme_x = None
    if len(batch) == 8:
        x, y, weights, tone_x, tone_y, rhyme_x, rhyme_y, theme_x = batch
    elif len(batch) == 7:
        x, y, tone_x, tone_y, rhyme_x, rhyme_y, theme_x = batch
    else:
        raise ValueError(f"未知 batch 结构: len={len(batch)}")
    x = x.to(device)
    y = y.to(device)
    tone_x = tone_x.to(device)
    tone_y = tone_y.to(device)
    rhyme_x = rhyme_x.to(device)
    rhyme_y = rhyme_y.to(device)
    theme_x = theme_x.to(device)
    if weights is not None:
        weights = weights.to(device)
    return x, y, weights, tone_x, tone_y, rhyme_x, rhyme_y, theme_x


def build_forward_kwargs(weights, tone_x, tone_y, rhyme_x, rhyme_y, theme_x):
    kwargs = {
        "tone_ids": tone_x,
        "rhyme_ids": rhyme_x,
        "tone_targets": tone_y,
        "rhyme_targets": rhyme_y,
        "theme_ids": theme_x,
    }
    if weights is not None:
        kwargs["loss_weights"] = weights
    return kwargs
# ======== Theme LoRA Batch Parser: END ========


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x, y, w, tone_x, tone_y, rhyme_x, rhyme_y, theme_x = unpack_theme_batch(batch, device)
        _, loss = model(x, y, **build_forward_kwargs(w, tone_x, tone_y, rhyme_x, rhyme_y, theme_x))
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


def load_theme_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    parser = argparse.ArgumentParser(description="主题 LoRA 微调")
    parser.add_argument("--train_pt", type=str, default=TRAIN_PT)
    parser.add_argument("--val_pt", type=str, default=VAL_PT)
    parser.add_argument("--train_tone_pt", type=str, default=TRAIN_TONE_PT)
    parser.add_argument("--val_tone_pt", type=str, default=VAL_TONE_PT)
    parser.add_argument("--train_rhyme_pt", type=str, default=TRAIN_RHYME_PT)
    parser.add_argument("--val_rhyme_pt", type=str, default=VAL_RHYME_PT)
    parser.add_argument("--train_theme_pt", type=str, default=TRAIN_THEME_PT)
    parser.add_argument("--val_theme_pt", type=str, default=VAL_THEME_PT)
    parser.add_argument("--theme_vocab", type=str, default=THEME_VOCAB)
    parser.add_argument("--vocab", type=str, default=VOCAB)
    parser.add_argument("--base_ckpt", type=str, default=BASE_CKPT)
    parser.add_argument("--save", type=str, default=SAVE_CKPT)
    parser.add_argument("--log_plot", type=str, default=LOSS_PLOT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--train_num_samples", type=int, default=TRAIN_NUM_SAMPLES)
    parser.add_argument("--val_batches", type=int, default=VAL_BATCHES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lora_rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora_alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    args = parser.parse_args()

    required = [
        args.train_pt,
        args.val_pt,
        args.train_tone_pt,
        args.val_tone_pt,
        args.train_rhyme_pt,
        args.val_rhyme_pt,
        args.train_theme_pt,
        args.val_theme_pt,
        args.theme_vocab,
        args.vocab,
        args.base_ckpt,
    ]
    for path in required:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    set_seed(args.seed)
    device = get_device()
    print("设备:", device, flush=True)

    stoi, _, vocab_size = load_vocab_json(args.vocab)
    theme_vocab = load_theme_vocab(args.theme_vocab)
    num_themes = int(theme_vocab["theme_count"]) + 1
    form_weight_map = build_loss_weight_by_token_id(
        stoi,
        line_sep_weight=LINE_SEP_WEIGHT,
        stanza_sep_weight=STANZA_SEP_WEIGHT,
        eos_weight=EOS_WEIGHT,
    )

    train_n = int(args.train_num_samples) if int(args.train_num_samples) > 0 else None
    train_ds = PoetryBlockDataset(
        args.train_pt,
        args.block_size,
        num_samples=train_n,
        sample_random=True,
        target_token_weights=form_weight_map,
        tone_path=args.train_tone_pt,
        rhyme_path=args.train_rhyme_pt,
        theme_path=args.train_theme_pt,
    )
    val_ds = PoetryBlockDataset(
        args.val_pt,
        args.block_size,
        num_samples=None,
        sample_random=False,
        target_token_weights=form_weight_map,
        tone_path=args.val_tone_pt,
        rhyme_path=args.val_rhyme_pt,
        theme_path=args.val_theme_pt,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    base_pack = torch.load(args.base_ckpt, map_location=device)
    base_hp = base_pack["hparams"]
    model = CharGPT(
        vocab_size=vocab_size,
        block_size=int(base_hp["block_size"]),
        d_model=int(base_hp["d_model"]),
        n_head=int(base_hp["n_head"]),
        n_layer=int(base_hp["n_layer"]),
        d_ff=int(base_hp["d_ff"]),
        dropout=float(base_hp.get("dropout", 0.1)),
        use_prosody=bool(base_hp.get("use_prosody", True)),
        num_tones=int(base_hp.get("num_tones", NUM_TONES)),
        num_rhymes=int(base_hp.get("num_rhymes", NUM_RHYMES + 1)),
        use_aux_loss=bool(base_hp.get("use_aux_loss", True)),
        aux_loss_weight=float(base_hp.get("aux_loss_weight", 0.1)),
        use_theme=True,
        num_themes=num_themes,
    ).to(device)
    missing, unexpected = model.load_state_dict(base_pack["model"], strict=False)
    allowed_missing = {"theme_emb.weight"}
    if set(missing) - allowed_missing:
        raise RuntimeError(f"加载底座 checkpoint 时缺少异常参数: {missing}")
    if unexpected:
        raise RuntimeError(f"加载底座 checkpoint 时出现多余参数: {unexpected}")
    model.enable_lora(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.freeze_base_for_theme_lora()

    trainable, total = count_trainable_params(model)
    print(
        f"主题数={theme_vocab['theme_count']} (+pad=1), "
        f"LoRA rank={args.lora_rank}, trainable={trainable:,}/{total:,}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore

    best_val = float("inf")
    train_losses: List[float] = []
    val_losses: List[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, count = 0.0, 0
        iterator: Iterable = train_loader
        if tqdm is not None:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", file=sys.stdout)
        for batch in iterator:
            x, y, w, tone_x, tone_y, rhyme_x, rhyme_y, theme_x = unpack_theme_batch(batch, device)
            optimizer.zero_grad()
            _, loss = model(x, y, **build_forward_kwargs(w, tone_x, tone_y, rhyme_x, rhyme_y, theme_x))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            running += loss.item() * x.size(0)
            count += x.size(0)
        train_loss = running / max(count, 1)
        val_loss = eval_loss(model, val_loader, device, max_batches=int(args.val_batches))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "hparams": {
                        "vocab_size": vocab_size,
                        "block_size": int(base_hp["block_size"]),
                        "d_model": int(base_hp["d_model"]),
                        "n_head": int(base_hp["n_head"]),
                        "n_layer": int(base_hp["n_layer"]),
                        "d_ff": int(base_hp["d_ff"]),
                        "dropout": float(base_hp.get("dropout", 0.1)),
                        "use_prosody": bool(base_hp.get("use_prosody", True)),
                        "num_tones": int(base_hp.get("num_tones", NUM_TONES)),
                        "num_rhymes": int(base_hp.get("num_rhymes", NUM_RHYMES + 1)),
                        "use_aux_loss": bool(base_hp.get("use_aux_loss", True)),
                        "aux_loss_weight": float(base_hp.get("aux_loss_weight", 0.1)),
                        "use_theme": True,
                        "num_themes": num_themes,
                        "use_lora": True,
                        "lora_rank": int(args.lora_rank),
                        "lora_alpha": float(args.lora_alpha),
                        "lora_dropout": float(args.lora_dropout),
                        "theme_vocab_path": args.theme_vocab,
                    },
                },
                args.save,
            )
            print(f"  -> 保存更优模型 val_loss={val_loss:.4f} 至 {args.save}", flush=True)

    if train_losses and val_losses:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.log_plot, dpi=150)
        print("已保存曲线:", os.path.abspath(args.log_plot), flush=True)
    print(f"主题 LoRA 微调结束。最佳 val_loss={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
