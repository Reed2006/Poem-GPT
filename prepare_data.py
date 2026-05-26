#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
古诗数据集下载与预处理脚本
- 从 Hugging Face 下载真实唐宋古诗数据
- 随机抽样最多 100000 条并保存为 poetry.txt
- 构建字符级词表并保存 vocab.json
- 划分 9:1 训练/验证集，编码后保存为 train_data.pt / val_data.pt
"""

import os
import json
import torch
from datasets import load_dataset

# 使用国内镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():
    print("=" * 50)
    print("步骤 1：下载真实唐宋古诗数据集")
    print("=" * 50)
    dataset = load_dataset("Lifan-Z/Chinese-poetries-txt", split="train")
    print(f"数据集总条数: {len(dataset)}")
    print(f"列名: {dataset.column_names}")

    # 随机打乱，避免只取前面数据带来的偏差
    dataset = dataset.shuffle(seed=42)

    # 确定存放诗词文本的列名（常见为 text 或 content）
    text_col = "text" if "text" in dataset.column_names else dataset.column_names[0]
    print(f"使用列: {text_col}")

    print("\n" + "=" * 50)
    print("步骤 2：随机抽样最多 100000 条并保存为 poetry.txt")
    print("=" * 50)
    n_samples = 100000
    actual_samples = min(n_samples, len(dataset))
    subset = dataset.select(range(actual_samples))

    # 每首诗占一行，用换行符拼接成一大段文本
    lines = []
    for item in subset:
        text = item[text_col]
        if text is None:
            continue
        text = str(text).strip()
        if text:
            # 去掉内部多余换行，避免一首诗被拆成多行
            text = text.replace("\r", "").replace("\n", "")
            lines.append(text)

    full_text = "\n".join(lines)

    with open("poetry.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"实际抽样条数: {len(lines)}")
    print(f"已保存 poetry.txt，总字符数约: {len(full_text)}")

    print("\n" + "=" * 50)
    print("步骤 3：构建字符级词表并保存 vocab.json")
    print("=" * 50)
    chars = sorted(list(set(full_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    vocab = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
    }

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"词表大小: {vocab_size}")

    print("\n" + "=" * 50)
    print("步骤 4：划分 9:1 训练/验证集并保存 .pt 文件")
    print("=" * 50)

    # 按“首”划分
    poems = [s.strip() for s in full_text.split("\n") if s.strip()]
    n_train = int(len(poems) * 0.9)

    train_poems = poems[:n_train]
    val_poems = poems[n_train:]

    train_text = "\n".join(train_poems)
    val_text = "\n".join(val_poems)

    train_ids = [stoi[c] for c in train_text]
    val_ids = [stoi[c] for c in val_text]

    train_data = torch.tensor(train_ids, dtype=torch.long)
    val_data = torch.tensor(val_ids, dtype=torch.long)

    torch.save(train_data, "train_data.pt")
    torch.save(val_data, "val_data.pt")

    print(f"训练集诗数: {len(train_poems)}")
    print(f"验证集诗数: {len(val_poems)}")
    print(f"训练集 token 数: {len(train_ids)}")
    print(f"验证集 token 数: {len(val_ids)}")
    print("已保存 train_data.pt 与 val_data.pt")

    print("\n预处理全部完成。")


if __name__ == "__main__":
    main()