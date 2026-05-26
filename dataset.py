# -*- coding: utf-8 -*-
"""
从 train_data.pt / val_data.pt 长序列中随机取连续块，构造 (x, y) 下一字预测任务。
x[i] 预测 y[i] = 序列中下一字符 id，长度均为 block_size。
"""
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def load_vocab_json(path: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi: Dict[str, int] = data["stoi"]
    itos: Dict[int, str] = {}
    for k, v in data["itos"].items():
        itos[int(k) if isinstance(k, str) else k] = v
    vs = int(data.get("vocab_size", len(itos)))
    return stoi, itos, vs


class PoetryBlockDataset(Dataset):
    """
    一维长序列上，以起始下标 i 取 [i : i+block_size] 为 x，
    [i+1 : i+1+block_size] 为 y（与实验指导 4.2 一致）。

    num_samples 为 None 时，每个合法起始下标 0..N-block-1 对应一个滑窗，长度可达千万，单 epoch 极慢。
    训练时建议设 num_samples 为有限值，并在 __getitem__ 中随机起窗（与常见 LM 训法一致）。
    """

    def __init__(
        self,
        data_path: str,
        block_size: int,
        num_samples: Optional[int] = None,
        sample_random: bool = True,
    ) -> None:
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)
        self.data = torch.load(data_path, map_location="cpu")
        if self.data.dim() != 1:
            raise ValueError("期望一维 long 张量")
        self.block_size = int(block_size)
        n = int(self.data.size(0))
        if n < self.block_size + 1:
            raise ValueError(f"序列太短: {n}，需 > block_size+1")
        self._max_i = n - self.block_size
        if num_samples is not None and int(num_samples) > 0:
            self._len = min(int(num_samples), self._max_i)
        else:
            self._len = int(self._max_i)
        # 显式子采样时：在长序列上随机起窗，避免一个 epoch 扫完全部滑窗
        self._sample_random = (
            bool(sample_random)
            and (num_samples is not None and int(num_samples) > 0)
            and (self._len < self._max_i)
        )

    def __len__(self) -> int:
        return int(self._len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._sample_random:
            i = random.randrange(0, self._max_i)
        else:
            i = int(idx) % int(self._max_i)
        x = self.data[i : i + self.block_size].clone()
        y = self.data[i + 1 : i + 1 + self.block_size].clone()
        return x, y
