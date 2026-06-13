#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path


BASE_DIR = Path("/Users/air/Documents/PJ9_Transformer/New Data")
INPUT_FILES = [
    BASE_DIR / "big_tang.train.txt",
    BASE_DIR / "big_tang.test.txt",
    BASE_DIR / "big_tang.val.txt",
]
OUTPUT_FILE = Path("/Users/air/Documents/PJ9_Transformer/new_data_2179_prefixed.txt")

CLAUSE_SPLIT_RE = re.compile(r"[，。！？；]")


def classify_poem(poem_text: str) -> str | None:
    clauses = [part.strip() for part in CLAUSE_SPLIT_RE.split(poem_text) if part.strip()]
    if not clauses:
        return None
    lengths = [len(clause) for clause in clauses]
    if len(set(lengths)) != 1:
        return None
    if len(clauses) == 4 and lengths[0] == 5:
        return "五言"
    if len(clauses) == 4 and lengths[0] == 7:
        return "七言"
    if len(clauses) == 8 and lengths[0] == 5:
        return "五言"
    if len(clauses) == 8 and lengths[0] == 7:
        return "七言"
    return None


def main() -> None:
    output_lines: list[str] = []
    for input_file in INPUT_FILES:
        with input_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.rstrip("\n")
                if not raw_line:
                    continue
                parts = raw_line.split("\t")
                if len(parts) < 2:
                    continue

                theme = parts[0].strip()
                poem_field = parts[1].strip()
                poem_text = poem_field.split("|", 1)[1].strip() if "|" in poem_field else poem_field
                yan_type = classify_poem(poem_text)
                if yan_type is None:
                    continue

                output_lines.append(f"【{yan_type}】【{theme}】{poem_text}")

    OUTPUT_FILE.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"总诗数: {len(output_lines)}")


if __name__ == "__main__":
    main()
