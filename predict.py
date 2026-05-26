# -*- coding: utf-8 -*-
"""
自回归续写 / 生成（与实验指导中「generate」对应；本文件命名为 predict.py 便于与 train 成对使用）。
从 checkpoint 加载 CharGPT，以**一个起始字**为条件自回归续写（默认只取你输入中落在词表里的**首字**为种子）；
用 temperature + multinomial 逐字采样。可用 --full_prompt 保留「整段作上下文」行为。

默认：交互式输入起笔字、输出续写；使用 --once 为一次性非交互（便于脚本）。
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from dataset import load_vocab_json
from model import CharGPT
from train import get_device


@torch.no_grad()
def sample_next(
    model: CharGPT, idx: torch.Tensor, temperature: float, device: torch.device
) -> int:
    """idx: (1, t)，取**最后一个**时间步的 logits 采样下一 token。"""
    model.eval()
    if idx.size(1) == 0:
        raise ValueError("序列为空")
    # 超长只保留最后 block_size
    t = min(idx.size(1), model.block_size)
    x = idx[:, -t:].contiguous()
    logits, _ = model(x)  # (1, t, V)
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


def first_in_vocab_char(s: str, stoi: Dict[str, int]) -> str:
    """
    仅将输入中**第一个**在词表内的字作为「起笔」条件，不把整段话当作多字上文。
    若你输入了多个字，只取首字；若仅一字则静默使用该字。
    """
    t = s.strip()
    if not t:
        raise ValueError("起笔为空")
    for ch in t:
        if ch in stoi:
            if len(t) > 1:
                print(f"（以首字「{ch}」为起笔；其余字不作为上文）", flush=True)
            return ch
    raise ValueError("输入中无在词表内的字，请另试")


def encode_prompt(prompt: str, stoi: Dict[str, int], device: torch.device) -> torch.Tensor:
    return _encode_chinese_prefix(prompt, stoi, device)


@torch.no_grad()
def generate_one(
    model: CharGPT,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    prompt: str,
    max_new: int,
    temperature: float,
    stop_newline: bool,
) -> str:
    """返回「起笔 + 续写」的完整串（与常见展示一致，起笔为单字时即一字 + 新采样）。"""
    idx = encode_prompt(prompt, stoi, device)
    out_ids: List[int] = idx[0].tolist()
    newline_id = stoi.get("\n", None)
    for _ in range(max_new):
        nxt = sample_next(
            model,
            torch.tensor([out_ids], device=device, dtype=torch.long),
            temperature,
            device,
        )
        out_ids.append(nxt)
        if stop_newline and newline_id is not None and nxt == newline_id:
            break
    return "".join(itos.get(i, "?") for i in out_ids)


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
    ).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, hp


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="古诗字符级自回归续写")
    p.add_argument("--ckpt", type=str, default="ckpt_best.pt")
    p.add_argument("--vocab", type=str, default="vocab.json")
    p.add_argument(
        "--once",
        action="store_true",
        help="非交互：仅用下方参数生成一次后退出；不设则进入交互式循环",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="春",
        help="--once 时：默认用首字为起笔；见 --full_prompt。交互时空行时的默认起笔",
    )
    p.add_argument(
        "--full_prompt",
        action="store_true",
        help="将整段输入/参数作为多字上文（原行为）；默认关闭，仅用词表内首字为起始字",
    )
    p.add_argument("--max_new", type=int, default=200, help="新增长度（字符数）")
    p.add_argument("--temperature", type=float, default=0.9, help=">0，越大越随机")
    p.add_argument(
        "--stop_newline",
        action="store_true",
        help="若采样到换行符 id 则提前停止",
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

    def run_one_line(user_line: str) -> None:
        raw = (user_line or "").strip() or str(args.prompt).strip()
        try:
            if args.full_prompt:
                p = raw
            else:
                p = first_in_vocab_char(raw, stoi)
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
                args.max_new,
                args.temperature,
                args.stop_newline,
            )
        except ValueError as e:
            print(e, file=sys.stderr)
            return
        print("续写:\n" + text + "\n", flush=True)

    if args.once:
        run_one_line("")
        return

    print(
        "交互续写：输入只作为起笔字（在词表中的首字为种子；多字时仅用第一字，见行内提示）。\n"
        "回车使用 --prompt 默认起笔「%s」；\n"
        "需要整段作上文时请加参数 --full_prompt 再启动。\n"
        "输入 q / quit / exit 结束。\n"
        "max_new=%d, temperature=%.2f\n"
        % (args.prompt, args.max_new, args.temperature)
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
