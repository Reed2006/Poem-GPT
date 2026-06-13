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
        target_token_weights: Optional[Dict[int, float]] = None,
        tone_path: Optional[str] = None,
        rhyme_path: Optional[str] = None,
        theme_path: Optional[str] = None,
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
        # ======== Form Weighting Dataset Hook: START ========
        # 变更说明:
        # - 允许按目标 token 生成逐位置 loss 权重。
        # - 这样 train.py 可以直接对行分隔符/阕分隔符/结束符施加更高监督强度。
        self.target_token_weights = {
            int(token_id): float(weight)
            for token_id, weight in (target_token_weights or {}).items()
            if float(weight) != 1.0
        }
        self.return_loss_weights = bool(self.target_token_weights)
        # ======== Form Weighting Dataset Hook: END ========
        # ======== Prosody Dataset Hook: START ========
        # 变更说明:
        # - 可选加载与 token 严格对齐的 tone/rhyme 长序列。
        # - __getitem__ 返回与 x/y 同窗口对齐的输入标签与目标标签。
        self.tone = None
        self.rhyme = None
        if tone_path:
            if not os.path.isfile(tone_path):
                raise FileNotFoundError(tone_path)
            self.tone = torch.load(tone_path, map_location="cpu")
            if self.tone.shape != self.data.shape:
                raise ValueError("tone 张量长度与 token 数据不一致")
        if rhyme_path:
            if not os.path.isfile(rhyme_path):
                raise FileNotFoundError(rhyme_path)
            self.rhyme = torch.load(rhyme_path, map_location="cpu")
            if self.rhyme.shape != self.data.shape:
                raise ValueError("rhyme 张量长度与 token 数据不一致")
        self.return_prosody = self.tone is not None and self.rhyme is not None
        self.theme = None
        if theme_path:
            if not os.path.isfile(theme_path):
                raise FileNotFoundError(theme_path)
            self.theme = torch.load(theme_path, map_location="cpu")
            if self.theme.shape != self.data.shape:
                raise ValueError("theme 张量长度与 token 数据不一致")
        self.return_theme = self.theme is not None
        # ======== Prosody Dataset Hook: END ========

    def __len__(self) -> int:
        return int(self._len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self._sample_random:
            i = random.randrange(0, self._max_i)
        else:
            i = int(idx) % int(self._max_i)
        x = self.data[i : i + self.block_size].clone()
        y = self.data[i + 1 : i + 1 + self.block_size].clone()
        tone_x = tone_y = rhyme_x = rhyme_y = None
        theme_x = None
        if self.return_prosody:
            tone_x = self.tone[i : i + self.block_size].clone()
            tone_y = self.tone[i + 1 : i + 1 + self.block_size].clone()
            rhyme_x = self.rhyme[i : i + self.block_size].clone()
            rhyme_y = self.rhyme[i + 1 : i + 1 + self.block_size].clone()
        if self.return_theme:
            theme_x = self.theme[i : i + self.block_size].clone()
        # ======== Form Weighting Dataset Hook: START ========
        # 变更说明:
        # - 对应目标位置默认权重为 1。
        # - 若该位置目标 token 是结构标记，则提升其 loss 权重。
        if self.return_loss_weights:
            weights = torch.ones_like(y, dtype=torch.float32)
            for token_id, weight in self.target_token_weights.items():
                weights[y == token_id] = float(weight)
            if self.return_prosody and self.return_theme:
                return x, y, weights, tone_x, tone_y, rhyme_x, rhyme_y, theme_x
            if self.return_prosody:
                return x, y, weights, tone_x, tone_y, rhyme_x, rhyme_y
            if self.return_theme:
                return x, y, weights, theme_x
            return x, y, weights
        # ======== Form Weighting Dataset Hook: END ========
        if self.return_prosody and self.return_theme:
            return x, y, tone_x, tone_y, rhyme_x, rhyme_y, theme_x
        if self.return_prosody:
            return x, y, tone_x, tone_y, rhyme_x, rhyme_y
        if self.return_theme:
            return x, y, theme_x
        return x, y
