# -*- coding: utf-8 -*-
"""
将四类校正结果合并为新的结构化训练文本，并编码成词表与 .pt 文件。

输入默认来自:
  processed_poetry_full/*.corrected.jsonl
  processed_poetry_full/*.errors.jsonl

输出默认包括:
  structured_poetry.txt
  structured_vocab.json
  structured_train_data.pt
  structured_val_data.pt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from poetry_format import GENRE_ORDER, format_structured_poem
from prosody import build_label_sequences


@dataclass
class StructuredPoemRecord:
    poem_id: int
    genre: str
    poem_text: str
    source: str


def load_genre_records(input_dir: Path, genre: str, include_error_fallback: bool) -> List[StructuredPoemRecord]:
    corrected_path = input_dir / f"{genre}.corrected.jsonl"
    errors_path = input_dir / f"{genre}.errors.jsonl"
    merged: Dict[int, StructuredPoemRecord] = {}

    if corrected_path.exists():
        with corrected_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                row = json.loads(raw_line)
                poem_id = int(row["poem_id"])
                merged[poem_id] = StructuredPoemRecord(
                    poem_id=poem_id,
                    genre=genre,
                    poem_text=str(row["corrected_poem"]).strip(),
                    source="corrected",
                )

    if include_error_fallback and errors_path.exists():
        with errors_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                row = json.loads(raw_line)
                poem_id = int(row["poem_id"])
                if poem_id in merged:
                    continue
                merged[poem_id] = StructuredPoemRecord(
                    poem_id=poem_id,
                    genre=genre,
                    poem_text=str(row["original_poem"]).strip(),
                    source="fallback_original",
                )

    return sorted(merged.values(), key=lambda item: item.poem_id)


def build_structured_lines(records: List[StructuredPoemRecord]) -> List[str]:
    lines: List[str] = []
    for record in sorted(records, key=lambda item: item.poem_id):
        lines.append(format_structured_poem(record.genre, record.poem_text))
    return lines


def build_vocab(full_text: str) -> Dict[str, object]:
    chars = sorted(list(set(full_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return {
        "vocab_size": len(chars),
        "stoi": stoi,
        "itos": itos,
    }


# ======== Prosody Dataset Export: START ========
# 变更说明:
# - 在原 token 序列之外，同步导出逐字符对齐的 tone/rhyme 张量。
# - 切分仍按“首”进行，保证结构化样本边界与旧版保持一致。
def encode_text(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def encode_prosody(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    tone_ids, rhyme_ids = build_label_sequences(text)
    return (
        torch.tensor(tone_ids, dtype=torch.long),
        torch.tensor(rhyme_ids, dtype=torch.long),
    )


def save_vocab_and_ids(
    full_text: str,
    vocab_path: Path,
    train_pt_path: Path,
    val_pt_path: Path,
    train_tone_path: Path,
    val_tone_path: Path,
    train_rhyme_path: Path,
    val_rhyme_path: Path,
    split_ratio: float,
) -> None:
    vocab = build_vocab(full_text)
    stoi = vocab["stoi"]
    with vocab_path.open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False, indent=2)

    poems = [s.strip() for s in full_text.split("\n") if s.strip()]
    n_train = int(len(poems) * split_ratio)
    train_text = "\n".join(poems[:n_train])
    val_text = "\n".join(poems[n_train:])

    train_ids = encode_text(train_text, stoi)
    val_ids = encode_text(val_text, stoi)
    train_tone, train_rhyme = encode_prosody(train_text)
    val_tone, val_rhyme = encode_prosody(val_text)

    if not (len(train_ids) == len(train_tone) == len(train_rhyme)):
        raise ValueError("训练集 token/tone/rhyme 未对齐")
    if not (len(val_ids) == len(val_tone) == len(val_rhyme)):
        raise ValueError("验证集 token/tone/rhyme 未对齐")

    torch.save(train_ids, train_pt_path)
    torch.save(val_ids, val_pt_path)
    torch.save(train_tone, train_tone_path)
    torch.save(val_tone, val_tone_path)
    torch.save(train_rhyme, train_rhyme_path)
    torch.save(val_rhyme, val_rhyme_path)
# ======== Prosody Dataset Export: END ========


def save_stats(records: List[StructuredPoemRecord], output_path: Path) -> None:
    stats: Dict[str, Dict[str, int]] = {}
    for genre in GENRE_ORDER:
        genre_records = [item for item in records if item.genre == genre]
        fallback_count = sum(1 for item in genre_records if item.source == "fallback_original")
        stats[genre] = {
            "total": len(genre_records),
            "corrected": sum(1 for item in genre_records if item.source == "corrected"),
            "fallback_original": fallback_count,
        }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于校正结果构建结构化古诗训练数据")
    parser.add_argument("--input-dir", type=str, default="processed_poetry_full")
    parser.add_argument("--output-txt", type=str, default="structured_poetry.txt")
    parser.add_argument("--output-vocab", type=str, default="structured_vocab.json")
    parser.add_argument("--output-train-pt", type=str, default="structured_train_data.pt")
    parser.add_argument("--output-val-pt", type=str, default="structured_val_data.pt")
    parser.add_argument("--output-train-tone-pt", type=str, default="structured_train_tone.pt")
    parser.add_argument("--output-val-tone-pt", type=str, default="structured_val_tone.pt")
    parser.add_argument("--output-train-rhyme-pt", type=str, default="structured_train_rhyme.pt")
    parser.add_argument("--output-val-rhyme-pt", type=str, default="structured_val_rhyme.pt")
    parser.add_argument("--output-stats", type=str, default="structured_poetry_stats.json")
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument(
        "--include-error-fallback",
        action="store_true",
        help="将校正失败样本回退为 original_poem 并纳入结构化数据；默认关闭，即剔除失败样本",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    os.chdir(here)

    input_dir = (here / args.input_dir).resolve()
    output_txt = (here / args.output_txt).resolve()
    output_vocab = (here / args.output_vocab).resolve()
    output_train_pt = (here / args.output_train_pt).resolve()
    output_val_pt = (here / args.output_val_pt).resolve()
    output_train_tone_pt = (here / args.output_train_tone_pt).resolve()
    output_val_tone_pt = (here / args.output_val_tone_pt).resolve()
    output_train_rhyme_pt = (here / args.output_train_rhyme_pt).resolve()
    output_val_rhyme_pt = (here / args.output_val_rhyme_pt).resolve()
    output_stats = (here / args.output_stats).resolve()

    all_records: List[StructuredPoemRecord] = []
    for genre in GENRE_ORDER:
        all_records.extend(load_genre_records(input_dir, genre, include_error_fallback=args.include_error_fallback))

    all_records.sort(key=lambda item: item.poem_id)
    structured_lines = build_structured_lines(all_records)
    full_text = "\n".join(structured_lines)

    with output_txt.open("w", encoding="utf-8") as handle:
        handle.write(full_text)

    save_vocab_and_ids(
        full_text=full_text,
        vocab_path=output_vocab,
        train_pt_path=output_train_pt,
        val_pt_path=output_val_pt,
        train_tone_path=output_train_tone_pt,
        val_tone_path=output_val_tone_pt,
        train_rhyme_path=output_train_rhyme_pt,
        val_rhyme_path=output_val_rhyme_pt,
        split_ratio=float(args.train_split),
    )
    save_stats(all_records, output_stats)

    print(f"结构化诗歌总数: {len(all_records)}")
    print(f"已保存结构化文本: {output_txt}")
    print(f"已保存词表: {output_vocab}")
    print(f"已保存训练集: {output_train_pt}")
    print(f"已保存验证集: {output_val_pt}")
    print(f"已保存训练集 Tone: {output_train_tone_pt}")
    print(f"已保存验证集 Tone: {output_val_tone_pt}")
    print(f"已保存训练集 Rhyme: {output_train_rhyme_pt}")
    print(f"已保存验证集 Rhyme: {output_val_rhyme_pt}")
    print(f"已保存统计: {output_stats}")


if __name__ == "__main__":
    main()
