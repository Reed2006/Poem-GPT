# -*- coding: utf-8 -*-
"""
结构化古诗的平仄/韵部标注工具。

设计目标：
1. 仅保留 Tone 和 Rhyme 两类标签，供 embedding 与辅助损失使用。
2. 与结构化文本严格逐字符对齐，包括【体裁】前缀、行分隔符、阕分隔符、结束符与换行。
3. 不做解码时硬约束，只提供训练/推理阶段可选的韵律条件输入。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

# ======== Prosody Label Config: START ========
# 变更说明:
# - 仅保留 Tone / Rhyme 两条标签链。
# - 结构标记、标点、换行统一落到 tone_punct / rhyme_unk，避免污染正文韵律监督。
TONE_PING = 0
TONE_ZE = 1
TONE_UNK = 2
TONE_PUNCT = 3
NUM_TONES = 4

RHYME_GROUPS = [
    "一麻",
    "二波",
    "三皆",
    "四开",
    "五微",
    "六豪",
    "七尤",
    "八寒",
    "九文",
    "十唐",
    "十一庚",
    "十二齐",
    "十三支",
    "十四姑",
]
NUM_RHYMES = len(RHYME_GROUPS)
RHYME_UNK = NUM_RHYMES
# ======== Prosody Label Config: END ========

GROUP2ID = {name: idx for idx, name in enumerate(RHYME_GROUPS)}
PUNCT_ALL = set("，。！？；：、")
STRUCTURE_TOKENS = set("【】|/@\n")
SPECIAL_TOKENS = PUNCT_ALL | STRUCTURE_TOKENS

FINAL2GROUP = {
    "a": 0,
    "ia": 0,
    "ua": 0,
    "o": 1,
    "e": 1,
    "uo": 1,
    "io": 1,
    "ie": 2,
    "ve": 2,
    "ue": 2,
    "ai": 3,
    "uai": 3,
    "ei": 4,
    "ui": 4,
    "uei": 4,
    "ao": 5,
    "iao": 5,
    "ou": 6,
    "iu": 6,
    "iou": 6,
    "an": 7,
    "ian": 7,
    "uan": 7,
    "van": 7,
    "en": 8,
    "in": 8,
    "un": 8,
    "uen": 8,
    "vn": 8,
    "ang": 9,
    "iang": 9,
    "uang": 9,
    "eng": 10,
    "ing": 10,
    "ong": 10,
    "iong": 10,
    "ueng": 10,
    "i": 11,
    "v": 11,
    "er": 11,
    "u": 13,
}
ZHI_INITIALS = {"zh", "ch", "sh", "r", "z", "c", "s"}
RUSHENG_CHARS = set(
    "一七八十白百帛国谷哭屋福服伏幅"
    "竹烛足族独读毒急及即集级极疾"
    "节杰洁结竭截接石实食拾说学穴"
    "决绝觉掘缺发罚伐乏答达杂拔出"
    "突卒捉浊夹峡狭菊局橘直值殖植"
    "失湿秃熟塾习席袭黑切"
)

PINGSHUI_PATH = Path("pingshui.json")
PINGSHUI = json.loads(PINGSHUI_PATH.read_text(encoding="utf-8")) if PINGSHUI_PATH.exists() else {}

try:
    from pypinyin import Style, pinyin

    HAS_PYPINYIN = True
except Exception:  # pragma: no cover
    HAS_PYPINYIN = False

WARNED_NO_SOURCE = False


def _warn_no_source() -> None:
    global WARNED_NO_SOURCE
    if not WARNED_NO_SOURCE:
        print(
            "[prosody] 警告：未找到 pingshui.json，且 pypinyin 不可用；"
            "Tone/Rhyme 标签将退回未知值。"
        )
        WARNED_NO_SOURCE = True


def is_special_char(ch: str) -> bool:
    return ch in SPECIAL_TOKENS


@lru_cache(maxsize=None)
def char_to_tone(ch: str) -> int:
    if is_special_char(ch):
        return TONE_PUNCT
    info = PINGSHUI.get(ch)
    if info is not None:
        return TONE_PING if info.get("tone") == "平" else TONE_ZE
    if not HAS_PYPINYIN:
        _warn_no_source()
        return TONE_UNK
    if ch in RUSHENG_CHARS:
        return TONE_ZE
    syllables = pinyin(ch, style=Style.TONE3, strict=False)
    value = syllables[0][0] if syllables and syllables[0] else ""
    matched = re.search(r"[1-5]", value)
    if not matched:
        return TONE_UNK
    tone_digit = matched.group()
    if tone_digit in {"1", "2"}:
        return TONE_PING
    if tone_digit == "5":
        return TONE_UNK
    return TONE_ZE


@lru_cache(maxsize=None)
def char_to_rhyme(ch: str) -> int:
    if is_special_char(ch):
        return RHYME_UNK
    if not HAS_PYPINYIN:
        _warn_no_source()
        return RHYME_UNK
    finals = pinyin(ch, style=Style.FINALS, strict=True)
    final = finals[0][0] if finals and finals[0] else ""
    final = final.replace("ü", "v").replace("u:", "v")
    if not final:
        return RHYME_UNK
    if final == "i":
        initials = pinyin(ch, style=Style.INITIALS, strict=True)
        initial = initials[0][0] if initials and initials[0] else ""
        return 12 if initial in ZHI_INITIALS else 11
    return FINAL2GROUP.get(final, RHYME_UNK)


def _labels_for_char(ch: str, inside_prefix: bool) -> Tuple[int, int]:
    if inside_prefix or is_special_char(ch):
        return TONE_PUNCT, RHYME_UNK
    return char_to_tone(ch), char_to_rhyme(ch)


def build_label_sequences(text: str) -> Tuple[List[int], List[int]]:
    """为结构化文本逐字符构造 tone/rhyme 标签。"""
    tone_ids: List[int] = []
    rhyme_ids: List[int] = []
    inside_prefix = False
    for ch in text:
        if ch == "【":
            inside_prefix = True
        tone_id, rhyme_id = _labels_for_char(ch, inside_prefix=inside_prefix)
        tone_ids.append(tone_id)
        rhyme_ids.append(rhyme_id)
        if ch == "】":
            inside_prefix = False
    return tone_ids, rhyme_ids

