# -*- coding: utf-8 -*-
"""
结构化古诗格式定义。

设计目标：
1. 每首诗前加体裁前缀，如【七律】。
2. 明确标出行分隔符、阕分隔符、结束符，便于模型学习结构。
3. 兼容字符级建模，优先使用单字符结构标记，方便做 form-stressed weighting。
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence


GENRE_ORDER: Sequence[str] = ("五绝", "七绝", "五律", "七律")
GENRE_RULES: Dict[str, tuple[int, int]] = {
    "五绝": (4, 5),
    "七绝": (4, 7),
    "五律": (8, 5),
    "七律": (8, 7),
}

# ======== Structured Poetry Tokens: START ========
# 变更说明:
# - 体裁标记保持用户要求的【七律】样式。
# - 行分隔符/阕分隔符/结束符采用单字符，方便字符级模型学习与加权监督。
# - 对于近体诗，使用 2+2 / 4+4 的半篇分界，显式暴露整体结构。
FORM_PREFIX_TEMPLATE = "【{genre}】"
THEME_PREFIX_TEMPLATE = "【{theme}】"
LINE_SEP = "|"
STANZA_SEP = "/"
POEM_EOS = "@"
# ======== Structured Poetry Tokens: END ========

CLAUSE_SPLIT_RE = re.compile(r"[，。！？；]")


def split_poem_lines(poem_text: str) -> List[str]:
    return [part.strip() for part in CLAUSE_SPLIT_RE.split(poem_text.strip()) if part.strip()]


def format_theme(theme: str) -> str:
    cleaned = str(theme).strip()
    if not cleaned:
        raise ValueError("主题不能为空")
    return THEME_PREFIX_TEMPLATE.format(theme=cleaned)


def format_prefix(genre: str, theme: str | None = None) -> str:
    if genre not in GENRE_RULES:
        raise ValueError(f"未知体裁: {genre}")
    prefix = FORM_PREFIX_TEMPLATE.format(genre=genre)
    if theme is not None and str(theme).strip():
        prefix += format_theme(theme)
    return prefix


def format_structured_poem(genre: str, poem_text: str, theme: str | None = None) -> str:
    lines = split_poem_lines(poem_text)
    expected_lines, _expected_chars = GENRE_RULES[genre]
    if len(lines) != expected_lines:
        raise ValueError(f"{genre} 句数不匹配: {len(lines)} != {expected_lines}")

    split_index = len(lines) // 2
    parts: List[str] = [format_prefix(genre, theme=theme)]
    for index, line in enumerate(lines, start=1):
        parts.append(line)
        if index == len(lines):
            parts.append(POEM_EOS)
        elif index == split_index:
            parts.append(STANZA_SEP)
        else:
            parts.append(LINE_SEP)
    return "".join(parts)


def structured_to_pretty_text(text: str) -> str:
    out = text.replace(POEM_EOS, "")
    out = out.replace(STANZA_SEP, "\n")
    out = out.replace(LINE_SEP, "\n")
    return out.strip()


def build_loss_weight_by_token_id(
    stoi: Dict[str, int],
    line_sep_weight: float = 2.0,
    stanza_sep_weight: float = 2.0,
    eos_weight: float = 3.0,
) -> Dict[int, float]:
    weight_map: Dict[int, float] = {}
    if LINE_SEP in stoi:
        weight_map[int(stoi[LINE_SEP])] = float(line_sep_weight)
    if STANZA_SEP in stoi:
        weight_map[int(stoi[STANZA_SEP])] = float(stanza_sep_weight)
    if POEM_EOS in stoi:
        weight_map[int(stoi[POEM_EOS])] = float(eos_weight)
    return weight_map


def first_valid_chars(text: str, vocab_chars: Iterable[str]) -> str:
    allowed = set(vocab_chars)
    return "".join(ch for ch in text if ch in allowed)
