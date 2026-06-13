#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选四种近体诗，并调用 DeepSeek API 对每句做校正。

输出目录结构示例：
processed_poetry/
  五绝.raw.txt
  五绝.corrected.jsonl
  五绝.corrected.txt
  七绝.raw.txt
  ...

环境变量：
  DEEPSEEK_API_KEY=...

示例：
  python3 clean_poetry_data.py --input poetry.txt --output-dir processed_poetry --limit-per-genre 3
  python3 clean_poetry_data.py --input poetry.txt --output-dir processed_poetry --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


GENRE_RULES: Dict[str, Tuple[int, int]] = {
    "五绝": (4, 5),
    "七绝": (4, 7),
    "五律": (8, 5),
    "七律": (8, 7),
}

CLAUSE_SPLIT_RE = re.compile(r"[，。！？；]")
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"


@dataclass
class PoemRecord:
    poem_id: int
    genre: str
    original_poem: str
    lines: List[str]


def split_poem_lines(poem_text: str) -> List[str]:
    return [part.strip() for part in CLAUSE_SPLIT_RE.split(poem_text.strip()) if part.strip()]


def is_han_char(ch: str) -> bool:
    return unicodedata.name(ch, "").startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))


def is_all_han_line(line: str) -> bool:
    return bool(line) and all(is_han_char(ch) for ch in line)


def join_poem_lines(lines: Sequence[str]) -> str:
    parts: List[str] = []
    total = len(lines)
    for index, line in enumerate(lines, start=1):
        if index == total:
            parts.append(f"{line}。")
        elif index % 2 == 1:
            parts.append(f"{line}，")
        else:
            parts.append(f"{line}。")
    return "".join(parts)


def classify_poem(poem_text: str) -> Optional[Tuple[str, List[str]]]:
    lines = split_poem_lines(poem_text)
    if not lines:
        return None

    line_count = len(lines)
    if line_count not in (4, 8):
        return None

    clean_lengths = []
    for line in lines:
        if not is_all_han_line(line):
            return None
        clean_lengths.append(len(line))

    if len(set(clean_lengths)) != 1:
        return None

    char_count = clean_lengths[0]
    for genre, (expected_lines, expected_chars) in GENRE_RULES.items():
        if line_count == expected_lines and char_count == expected_chars:
            return genre, lines
    return None


def load_poems(input_path: Path) -> Dict[str, List[PoemRecord]]:
    grouped: Dict[str, List[PoemRecord]] = {genre: [] for genre in GENRE_RULES}
    poem_id = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            text = raw_line.strip()
            if not text:
                continue
            poem_id += 1
            classified = classify_poem(text)
            if not classified:
                continue
            genre, lines = classified
            grouped[genre].append(
                PoemRecord(
                    poem_id=poem_id,
                    genre=genre,
                    original_poem=text,
                    lines=lines,
                )
            )
    return grouped


def write_raw_files(grouped: Dict[str, List[PoemRecord]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for genre, poems in grouped.items():
        raw_path = output_dir / f"{genre}.raw.txt"
        with raw_path.open("w", encoding="utf-8") as handle:
            for poem in poems:
                handle.write(poem.original_poem + "\n")


def load_completed_poem_ids(jsonl_path: Path) -> set[int]:
    completed_ids: set[int] = set()
    if not jsonl_path.exists():
        return completed_ids

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            poem_id = row.get("poem_id")
            if isinstance(poem_id, int):
                completed_ids.add(poem_id)
    return completed_ids


class DeepSeekClient:
    def __init__(self, api_key: str, model: str, api_url: str, timeout: int, retries: int) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.retries = retries

    def correct_poem(self, poem: PoemRecord) -> dict:
        payload = {
            "model": self.model,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是严谨的中文古诗文本校对助手。"
                        "你只做尽量小的文本校正，不重写诗意，不扩写，不解释格律。"
                        "必须保持原句数不变、每句字数不变。"
                        "输出必须是 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": build_prompt(poem),
                },
            ],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 2048,
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            request = urllib.request.Request(
                self.api_url,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
                parsed = json.loads(response_body)
                content = parsed["choices"][0]["message"]["content"]
                result = json.loads(content)
                validated = validate_llm_result(poem, result)
                validated["usage"] = parsed.get("usage", {})
                return validated
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(f"DeepSeek 调用失败: {last_error}")


def build_prompt(poem: PoemRecord) -> str:
    lines_block = "\n".join(f"{idx}. {line}" for idx, line in enumerate(poem.lines, start=1))
    return (
        f"请校对下面这首{poem.genre}的每一句。\n"
        "要求：\n"
        "1. 每句只做必要校正，尽量保留原文。\n"
        "2. 必须保持句数不变。\n"
        "3. 每句修正后字数必须与原句完全一致。\n"
        "4. 对每一句输出：原句、修正后句子、原因、置信度。\n"
        "5. 置信度是 0 到 1 之间的小数。\n"
        "6. 如果无需修改，修正后句子就等于原句，并说明原因。\n"
        "7. 只输出 JSON，格式如下：\n"
        "{\n"
        '  "genre": "五绝/七绝/五律/七律",\n'
        '  "lines": [\n'
        "    {\n"
        '      "index": 1,\n'
        '      "original": "原句",\n'
        '      "corrected": "修正后句子",\n'
        '      "reason": "一句话原因",\n'
        '      "confidence": 0.95\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"诗歌内容：\n{lines_block}"
    )


def validate_llm_result(poem: PoemRecord, result: dict) -> dict:
    if result.get("genre") != poem.genre:
        raise ValueError(f"体裁不匹配: {result.get('genre')} != {poem.genre}")

    lines = result.get("lines")
    if not isinstance(lines, list) or len(lines) != len(poem.lines):
        raise ValueError("返回句数不匹配")

    normalized_lines = []
    corrected_poem_lines = []
    for expected_index, (original_line, row) in enumerate(zip(poem.lines, lines), start=1):
        if not isinstance(row, dict):
            raise ValueError("返回行格式错误")

        corrected = str(row.get("corrected", "")).strip()
        reason = str(row.get("reason", "")).strip()
        confidence = row.get("confidence")
        returned_original = str(row.get("original", "")).strip()
        returned_index = row.get("index")

        if returned_original and returned_original != original_line:
            raise ValueError(f"原句回显不一致: {returned_original} != {original_line}")
        if returned_index not in (None, expected_index):
            raise ValueError(f"句序不一致: {returned_index} != {expected_index}")
        if not corrected:
            raise ValueError("修正结果为空")
        if len(corrected) != len(original_line):
            raise ValueError(f"字数变化: {original_line} -> {corrected}")
        if not is_all_han_line(corrected):
            raise ValueError(f"修正结果包含非汉字字符: {corrected}")
        if not reason:
            raise ValueError("缺少原因")

        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"置信度格式错误: {confidence}") from exc

        if confidence_value < 0 or confidence_value > 1:
            raise ValueError(f"置信度超出范围: {confidence_value}")

        normalized_lines.append(
            {
                "index": expected_index,
                "original": original_line,
                "corrected": corrected,
                "reason": reason,
                "confidence": confidence_value,
            }
        )
        corrected_poem_lines.append(corrected)

    return {
        "genre": poem.genre,
        "lines": normalized_lines,
        "corrected_poem": join_poem_lines(corrected_poem_lines),
    }


def process_genre(
    genre: str,
    poems: Sequence[PoemRecord],
    client: DeepSeekClient,
    output_dir: Path,
    workers: int,
    limit: Optional[int],
) -> Tuple[int, int]:
    jsonl_path = output_dir / f"{genre}.corrected.jsonl"
    txt_path = output_dir / f"{genre}.corrected.txt"
    completed_ids = load_completed_poem_ids(jsonl_path)
    pending = [poem for poem in poems if poem.poem_id not in completed_ids]
    if limit is not None:
        pending = pending[:limit]

    lock = threading.Lock()
    success_count = 0
    failure_count = 0

    with jsonl_path.open("a", encoding="utf-8") as jsonl_handle, txt_path.open("a", encoding="utf-8") as txt_handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(client.correct_poem, poem): poem for poem in pending}
            for future in as_completed(future_map):
                poem = future_map[future]
                try:
                    result = future.result()
                    row = {
                        "poem_id": poem.poem_id,
                        "genre": poem.genre,
                        "original_poem": poem.original_poem,
                        "corrected_poem": result["corrected_poem"],
                        "lines": result["lines"],
                        "usage": result.get("usage", {}),
                    }
                    with lock:
                        jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        jsonl_handle.flush()
                        txt_handle.write(result["corrected_poem"] + "\n")
                        txt_handle.flush()
                    success_count += 1
                except Exception as exc:
                    failure_count += 1
                    error_row = {
                        "poem_id": poem.poem_id,
                        "genre": poem.genre,
                        "original_poem": poem.original_poem,
                        "error": str(exc),
                    }
                    error_path = output_dir / f"{genre}.errors.jsonl"
                    with lock:
                        with error_path.open("a", encoding="utf-8") as error_handle:
                            error_handle.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                    print(f"[{genre}] poem_id={poem.poem_id} 失败: {exc}")

    return success_count, failure_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选四种近体诗，并调用 DeepSeek 做逐句校正")
    parser.add_argument("--input", required=True, help="输入 poetry.txt 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="DeepSeek API URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek 模型名")
    parser.add_argument("--workers", type=int, default=4, help="并发请求数")
    parser.add_argument("--timeout", type=int, default=120, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=5, help="单次请求重试次数")
    parser.add_argument("--limit-per-genre", type=int, default=None, help="每个体裁最多处理多少首，用于测试")
    parser.add_argument("--skip-llm", action="store_true", help="只筛选落盘，不调用大模型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print("开始读取并筛选诗歌...")
    grouped = load_poems(input_path)
    write_raw_files(grouped, output_dir)

    total_kept = 0
    for genre in ("五绝", "七绝", "五律", "七律"):
        count = len(grouped[genre])
        total_kept += count
        print(f"{genre}: {count} 首")
    print(f"筛选后总数: {total_kept} 首")

    if args.skip_llm:
        print("已跳过大模型校正。")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")

    client = DeepSeekClient(
        api_key=api_key,
        model=args.model,
        api_url=args.api_url,
        timeout=args.timeout,
        retries=args.retries,
    )

    grand_success = 0
    grand_failure = 0
    for genre in ("五绝", "七绝", "五律", "七律"):
        print(f"开始处理 {genre} ...")
        success_count, failure_count = process_genre(
            genre=genre,
            poems=grouped[genre],
            client=client,
            output_dir=output_dir,
            workers=args.workers,
            limit=args.limit_per_genre,
        )
        grand_success += success_count
        grand_failure += failure_count
        print(f"{genre} 完成: 成功 {success_count}，失败 {failure_count}")

    print(f"全部完成: 成功 {grand_success}，失败 {grand_failure}")


if __name__ == "__main__":
    main()
