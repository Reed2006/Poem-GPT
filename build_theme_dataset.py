# -*- coding: utf-8 -*-
"""
基于带主题前缀的新数据，构建主题微调所需的结构化 token / tone / rhyme / theme 张量。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from poetry_format import format_structured_poem
from prosody import RHYME_UNK, TONE_PUNCT, build_label_sequences

INPUT_PATTERN = re.compile(r"^【([^】]+)】【([^】]+)】(.*)$")
CLAUSE_SPLIT_RE = re.compile(r"[，。！？；]")
YAN_TO_GENRE = {
    (4, 5): "五绝",
    (4, 7): "七绝",
    (8, 5): "五律",
    (8, 7): "七律",
}
OOV_NORMALIZE_MAP = {
    "壘": "垒",
    "嶮": "险",
    "廻": "回",
    "擬": "拟",
    "曨": "胧",
    "氾": "泛",
    "浕": "尽",
    "濛": "蒙",
    "瀰": "弥",
    "犠": "牺",
    "蘺": "篱",
    "褭": "袅",
    "覉": "羁",
    "遶": "绕",
    "醆": "盏",
    "鍊": "炼",
    "閒": "闲",
    "鞦": "秋",
    "鶬": "苍",
}


@dataclass
class ThemedPoemRecord:
    genre: str
    theme: str
    poem_text: str


def classify_genre(poem_text: str) -> str:
    clauses = [part.strip() for part in CLAUSE_SPLIT_RE.split(poem_text.strip()) if part.strip()]
    if not clauses:
        raise ValueError("空诗句")
    lengths = {len(clause) for clause in clauses}
    if len(lengths) != 1:
        raise ValueError(f"句长不一致: {poem_text}")
    key = (len(clauses), len(clauses[0]))
    if key not in YAN_TO_GENRE:
        raise ValueError(f"不支持的体裁: {key}")
    return YAN_TO_GENRE[key]


def load_records(input_path: Path) -> List[ThemedPoemRecord]:
    records: List[ThemedPoemRecord] = []
    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = INPUT_PATTERN.match(line)
        if not matched:
            raise ValueError(f"无法解析行: {line}")
        _yan, theme, poem_text = matched.groups()
        records.append(
            ThemedPoemRecord(
                genre=classify_genre(poem_text),
                theme=theme.strip(),
                poem_text=poem_text.strip(),
            )
        )
    return records


def normalize_poem_text(text: str) -> str:
    for src, dst in OOV_NORMALIZE_MAP.items():
        text = text.replace(src, dst)
    return text


def build_theme_vocab(records: Sequence[ThemedPoemRecord]) -> Dict[str, object]:
    themes = sorted({record.theme for record in records})
    stoi = {theme: idx for idx, theme in enumerate(themes)}
    pad_id = len(themes)
    return {
        "stoi": stoi,
        "itos": {str(idx): theme for theme, idx in stoi.items()},
        "theme_count": len(themes),
        "pad_id": pad_id,
    }


# ======== Theme Dataset Assembly: START ========
# 变更说明:
# - 结构化文本保留【体裁】【主题】前缀。
# - theme_id 作为逐位置标签导出，在整首诗范围内广播；诗与诗之间的换行用 pad_id。
def encode_records(
    records: Sequence[ThemedPoemRecord],
    vocab_stoi: Dict[str, int],
    theme_stoi: Dict[str, int],
    theme_pad_id: int,
) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    text_chunks: List[str] = []
    token_ids: List[int] = []
    tone_ids: List[int] = []
    rhyme_ids: List[int] = []
    theme_ids: List[int] = []

    for index, record in enumerate(records):
        structured = format_structured_poem(record.genre, record.poem_text, theme=record.theme)
        local_tone, local_rhyme = build_label_sequences(structured)
        local_tokens = [vocab_stoi[ch] for ch in structured]
        if not (len(local_tokens) == len(local_tone) == len(local_rhyme)):
            raise ValueError("结构化 token / tone / rhyme 未对齐")

        text_chunks.append(structured)
        token_ids.extend(local_tokens)
        tone_ids.extend(local_tone)
        rhyme_ids.extend(local_rhyme)
        theme_ids.extend([int(theme_stoi[record.theme])] * len(local_tokens))

        if index != len(records) - 1:
            text_chunks.append("\n")
            token_ids.append(int(vocab_stoi["\n"]))
            tone_ids.append(TONE_PUNCT)
            rhyme_ids.append(RHYME_UNK)
            theme_ids.append(int(theme_pad_id))

    full_text = "".join(text_chunks)
    return (
        full_text,
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(tone_ids, dtype=torch.long),
        torch.tensor(rhyme_ids, dtype=torch.long),
        torch.tensor(theme_ids, dtype=torch.long),
    )
# ======== Theme Dataset Assembly: END ========


def save_theme_stats(records: Sequence[ThemedPoemRecord], output_path: Path) -> None:
    genre_stats: Dict[str, int] = {}
    theme_stats: Dict[str, int] = {}
    for record in records:
        genre_stats[record.genre] = genre_stats.get(record.genre, 0) + 1
        theme_stats[record.theme] = theme_stats.get(record.theme, 0) + 1
    payload = {
        "total": len(records),
        "genres": genre_stats,
        "themes": theme_stats,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建主题 LoRA 微调数据")
    parser.add_argument("--input-txt", type=str, default="new_data_2179_prefixed.txt")
    parser.add_argument("--base-vocab", type=str, default="structured_vocab.json")
    parser.add_argument("--output-txt", type=str, default="theme_structured_poetry.txt")
    parser.add_argument("--output-theme-vocab", type=str, default="theme_vocab.json")
    parser.add_argument("--output-train-pt", type=str, default="theme_train_data.pt")
    parser.add_argument("--output-val-pt", type=str, default="theme_val_data.pt")
    parser.add_argument("--output-train-tone-pt", type=str, default="theme_train_tone.pt")
    parser.add_argument("--output-val-tone-pt", type=str, default="theme_val_tone.pt")
    parser.add_argument("--output-train-rhyme-pt", type=str, default="theme_train_rhyme.pt")
    parser.add_argument("--output-val-rhyme-pt", type=str, default="theme_val_rhyme.pt")
    parser.add_argument("--output-train-theme-pt", type=str, default="theme_train_theme.pt")
    parser.add_argument("--output-val-theme-pt", type=str, default="theme_val_theme.pt")
    parser.add_argument("--output-stats", type=str, default="theme_structured_stats.json")
    parser.add_argument("--train-split", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    os.chdir(here)

    input_txt = (here / args.input_txt).resolve()
    base_vocab_path = (here / args.base_vocab).resolve()
    output_txt = (here / args.output_txt).resolve()
    output_theme_vocab = (here / args.output_theme_vocab).resolve()
    output_train_pt = (here / args.output_train_pt).resolve()
    output_val_pt = (here / args.output_val_pt).resolve()
    output_train_tone_pt = (here / args.output_train_tone_pt).resolve()
    output_val_tone_pt = (here / args.output_val_tone_pt).resolve()
    output_train_rhyme_pt = (here / args.output_train_rhyme_pt).resolve()
    output_val_rhyme_pt = (here / args.output_val_rhyme_pt).resolve()
    output_train_theme_pt = (here / args.output_train_theme_pt).resolve()
    output_val_theme_pt = (here / args.output_val_theme_pt).resolve()
    output_stats = (here / args.output_stats).resolve()

    raw_records = load_records(input_txt)
    base_vocab = json.loads(base_vocab_path.read_text(encoding="utf-8"))
    vocab_stoi = base_vocab["stoi"]
    # ======== Theme OOV Cleanup: START ========
    # 变更说明:
    # - 主题数据沿用旧词表做增量微调，不能随意扩词表。
    # - 对常见异体/繁体字做最小归一化；仍无法落入词表的样本直接剔除。
    records: List[ThemedPoemRecord] = []
    dropped = 0
    dropped_chars: Dict[str, int] = {}
    for record in raw_records:
        normalized_text = normalize_poem_text(record.poem_text)
        normalized = ThemedPoemRecord(
            genre=record.genre,
            theme=record.theme,
            poem_text=normalized_text,
        )
        structured = format_structured_poem(normalized.genre, normalized.poem_text, theme=normalized.theme)
        missing = sorted({ch for ch in structured if ch not in vocab_stoi})
        if missing:
            dropped += 1
            for ch in missing:
                dropped_chars[ch] = dropped_chars.get(ch, 0) + 1
            continue
        records.append(normalized)
    # ======== Theme OOV Cleanup: END ========
    theme_vocab = build_theme_vocab(records)
    theme_stoi = theme_vocab["stoi"]
    theme_pad_id = int(theme_vocab["pad_id"])

    n_train = int(len(records) * float(args.train_split))
    train_records = records[:n_train]
    val_records = records[n_train:]

    train_text, train_ids, train_tone, train_rhyme, train_theme = encode_records(
        train_records, vocab_stoi, theme_stoi, theme_pad_id
    )
    val_text, val_ids, val_tone, val_rhyme, val_theme = encode_records(
        val_records, vocab_stoi, theme_stoi, theme_pad_id
    )

    output_txt.write_text(train_text + ("\n" if train_text and val_text else "") + val_text, encoding="utf-8")
    output_theme_vocab.write_text(json.dumps(theme_vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(train_ids, output_train_pt)
    torch.save(val_ids, output_val_pt)
    torch.save(train_tone, output_train_tone_pt)
    torch.save(val_tone, output_val_tone_pt)
    torch.save(train_rhyme, output_train_rhyme_pt)
    torch.save(val_rhyme, output_val_rhyme_pt)
    torch.save(train_theme, output_train_theme_pt)
    torch.save(val_theme, output_val_theme_pt)
    save_theme_stats(records, output_stats)

    print(f"主题数据总数: {len(records)}")
    print(f"OOV 归一化后剔除样本: {dropped}")
    print(f"训练/验证: {len(train_records)}/{len(val_records)}")
    print(f"主题数: {theme_vocab['theme_count']}")
    print(f"已保存主题词表: {output_theme_vocab}")
    if dropped_chars:
        print(f"仍无法映射的字符统计: {sorted(dropped_chars.items(), key=lambda item: (-item[1], item[0]))[:10]}")


if __name__ == "__main__":
    main()
