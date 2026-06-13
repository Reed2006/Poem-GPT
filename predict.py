# -*- coding: utf-8 -*-
"""
自回归续写 / 生成（与实验指导中「generate」对应；本文件命名为 predict.py 便于与 train 成对使用）。
从 checkpoint 加载 CharGPT，以“体裁 + 初字”为条件做结构化续写；
默认生成入口形如：【七律】春
并在结束符生成后停止。
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from dataset import load_vocab_json
from model import CharGPT
from poetry_format import GENRE_ORDER, POEM_EOS, format_prefix, structured_to_pretty_text
from prosody import build_label_sequences
from train import get_device


@torch.no_grad()
def sample_next(
    model: CharGPT,
    idx: torch.Tensor,
    temperature: float,
    device: torch.device,
    tone_ids: torch.Tensor | None = None,
    rhyme_ids: torch.Tensor | None = None,
    theme_ids: torch.Tensor | None = None,
) -> int:
    """idx: (1, t)，取**最后一个**时间步的 logits 采样下一 token。"""
    model.eval()
    if idx.size(1) == 0:
        raise ValueError("序列为空")
    # 超长只保留最后 block_size
    t = min(idx.size(1), model.block_size)
    x = idx[:, -t:].contiguous()
    kwargs = {}
    if model.use_prosody and tone_ids is not None and rhyme_ids is not None:
        kwargs["tone_ids"] = tone_ids[:, -t:].contiguous()
        kwargs["rhyme_ids"] = rhyme_ids[:, -t:].contiguous()
    if model.use_theme and theme_ids is not None:
        kwargs["theme_ids"] = theme_ids[:, -t:].contiguous()
    logits, _ = model(x, **kwargs)  # (1, t, V)
    last = logits[:, -1, :] / max(temperature, 1e-6)
    p = F.softmax(last, dim=-1)
    nxt = torch.multinomial(p, num_samples=1).item()
    return int(nxt)


def _encode_chinese_prefix(prefix: str, stoi: Dict[str, int], device: torch.device) -> torch.Tensor:
    """将 prefix 中每个字符编为 id；可跳过 OOV 字符。"""
    ids: List[int] = []
    for ch in prefix:
        if ch not in stoi:
            print(f"警告: 字符不在词表，已跳过: {repr(ch)}", file=sys.stderr)
            continue
        ids.append(int(stoi[ch]))
    if not ids:
        raise ValueError("编码后无有效字，请换起笔或检查词表")
    return torch.tensor([ids], dtype=torch.long, device=device)


def encode_prompt(prompt: str, stoi: Dict[str, int], device: torch.device) -> torch.Tensor:
    return _encode_chinese_prefix(prompt, stoi, device)


# ======== Prosody Inference Helper: START ========
# 变更说明:
# - 推理阶段不做硬约束，只按当前上下文动态补齐 Tone/Rhyme embedding。
# - 这样生成逻辑与旧版保持一致，只是额外把韵律信息喂给模型。
def encode_prosody_context(text: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tone_ids, rhyme_ids = build_label_sequences(text)
    return (
        torch.tensor([tone_ids], dtype=torch.long, device=device),
        torch.tensor([rhyme_ids], dtype=torch.long, device=device),
    )
# ======== Prosody Inference Helper: END ========


def load_theme_vocab(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@torch.no_grad()
def generate_one(
    model: CharGPT,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    prompt: str,
    theme_id: int | None,
    max_new: int,
    temperature: float,
    stop_eos: bool,
) -> str:
    """返回完整结构化文本，如【七律】春...@。"""
    idx = encode_prompt(prompt, stoi, device)
    out_ids: List[int] = idx[0].tolist()
    out_text = "".join(itos.get(i, "?") for i in out_ids)
    eos_id = stoi.get(POEM_EOS, None)
    for _ in range(max_new):
        tone_ids = rhyme_ids = None
        if model.use_prosody:
            tone_ids, rhyme_ids = encode_prosody_context(out_text, device)
        theme_ids = None
        if model.use_theme:
            if theme_id is None:
                raise ValueError("当前模型需要主题 id，但未提供 theme")
            theme_ids = torch.full((1, len(out_ids)), int(theme_id), dtype=torch.long, device=device)
        nxt = sample_next(
            model,
            torch.tensor([out_ids], device=device, dtype=torch.long),
            temperature,
            device,
            tone_ids=tone_ids,
            rhyme_ids=rhyme_ids,
            theme_ids=theme_ids,
        )
        out_ids.append(nxt)
        out_text += itos.get(nxt, "?")
        if stop_eos and eos_id is not None and nxt == eos_id:
            break
    return out_text


def load_for_generate(ckpt_path: str, device: torch.device) -> Tuple[CharGPT, dict]:
    pack = torch.load(ckpt_path, map_location=device)
    sd = pack["model"]
    hp = pack["hparams"]
    m = CharGPT(
        vocab_size=hp["vocab_size"],
        block_size=hp["block_size"],
        d_model=hp["d_model"],
        n_head=hp["n_head"],
        n_layer=hp["n_layer"],
        d_ff=hp["d_ff"],
        dropout=float(hp.get("dropout", 0.1)),
        use_prosody=bool(hp.get("use_prosody", False)),
        num_tones=int(hp.get("num_tones", 4)),
        num_rhymes=int(hp.get("num_rhymes", 15)),
        use_aux_loss=bool(hp.get("use_aux_loss", False)),
        aux_loss_weight=float(hp.get("aux_loss_weight", 0.1)),
        use_theme=bool(hp.get("use_theme", False)),
        num_themes=int(hp.get("num_themes", 0)),
        use_lora=bool(hp.get("use_lora", False)),
        lora_rank=int(hp.get("lora_rank", 8)),
        lora_alpha=float(hp.get("lora_alpha", 16.0)),
        lora_dropout=float(hp.get("lora_dropout", 0.0)),
    ).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, hp


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="古诗字符级自回归续写")
    p.add_argument("--ckpt", type=str, default="structured_ckpt_best.pt")
    p.add_argument("--vocab", type=str, default="structured_vocab.json")
    p.add_argument(
        "--once",
        action="store_true",
        help="非交互：仅用下方参数生成一次后退出；不设则进入交互式循环",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="春",
        help="起始字或起始短语；会接在体裁前缀后，形成【体裁】+起始文本",
    )
    p.add_argument(
        "--genre",
        type=str,
        default="七绝",
        choices=list(GENRE_ORDER),
        help="结构化生成时的体裁前缀",
    )
    p.add_argument("--theme", type=str, default="", help="可选主题，例如 战争 / 送别 / 山水")
    p.add_argument("--theme_vocab", type=str, default="theme_vocab.json", help="主题词表路径")
    p.add_argument(
        "--raw_prompt",
        action="store_true",
        help="直接把 --prompt 当作原始上下文，不自动补【体裁】前缀",
    )
    p.add_argument("--max_new", type=int, default=200, help="新增长度（字符数）")
    p.add_argument("--temperature", type=float, default=0.9, help=">0，越大越随机")
    p.add_argument(
        "--raw_output",
        action="store_true",
        help="打印原始结构化文本，不做美化展示",
    )
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        print("未找到:", args.ckpt, "请先训练: python train.py", file=sys.stderr)
        sys.exit(1)

    print("正在加载模型与词表…", flush=True)
    stoi, itos, _ = load_vocab_json(args.vocab)
    device = get_device()
    model, _hp = load_for_generate(args.ckpt, device)
    print("设备:", device, flush=True)
    theme_id = None
    if model.use_theme:
        if not str(args.theme).strip():
            raise ValueError("当前 checkpoint 需要 --theme")
        theme_vocab = load_theme_vocab(args.theme_vocab)
        theme_stoi = theme_vocab["stoi"]
        if args.theme not in theme_stoi:
            raise ValueError(f"主题不在词表中: {args.theme}")
        theme_id = int(theme_stoi[args.theme])

    # ======== Structured Inference Prompt: START ========
    # 变更说明:
    # - 推理入口固定为“体裁 + 初字/起始短语”。
    # - 默认提示形如【七律】春，符合新的结构化训练数据格式。
    def build_generation_prompt(seed_text: str) -> str:
        cleaned_seed = seed_text.strip()
        if args.raw_prompt:
            raw = cleaned_seed
        else:
            raw = format_prefix(args.genre, theme=args.theme if model.use_theme else None) + cleaned_seed
        if not raw:
            raise ValueError("生成 prompt 为空")
        return raw
    # ======== Structured Inference Prompt: END ========

    def run_one_line(user_line: str) -> None:
        raw = (user_line or "").strip() or str(args.prompt).strip()
        try:
            p = build_generation_prompt(raw)
        except ValueError as e:
            print(e, file=sys.stderr)
            return
        try:
            text = generate_one(
                model,
                stoi,
                itos,
                device,
                p,
                theme_id,
                args.max_new,
                args.temperature,
                True,
            )
        except ValueError as e:
            print(e, file=sys.stderr)
            return
        display_text = text if args.raw_output else structured_to_pretty_text(text)
        print("续写:\n" + display_text + "\n", flush=True)

    if args.once:
        run_one_line("")
        return

    print(
        "交互续写：输入会接在体裁前缀后，形成结构化 prompt。\n"
        "当前模式: 【%s】+起始字/短语\n"
        "回车使用 --prompt 默认起笔「%s」；\n"
        "输入 q / quit / exit 结束。\n"
        "max_new=%d, temperature=%.2f\n"
        % (args.genre, args.prompt, args.max_new, args.temperature)
    )
    while True:
        try:
            line = input("起笔字> ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        s = line.strip()
        if s.lower() in ("q", "quit", "exit"):
            print("再见。")
            break
        if not s and not str(args.prompt).strip():
            print("起笔与默认 --prompt 均为空，请重试", file=sys.stderr)
            continue
        run_one_line(s)


if __name__ == "__main__":
    main()
