# -*- coding: utf-8 -*-
"""
字符级 Decoder-only 语言模型（GPT 式）。
用多头因果自注意力 + FFN 堆叠；**未使用** nn.Transformer 封装。

待补全项：CausalSelfAttention.forward 与 FeedForward（__init__ + forward），详见《实验指导》。
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model 应能被 n_head 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)，C == d_model。
        需实现：多头 Q/K/V → 缩放点积注意力 → 因果掩码（不能看到未来位置）→ softmax →
        与 V 相乘 → 合并多头 → 输出线性层。

        提示：
        - 注意力 logits 在 **d_head** 维度上按 sqrt(d_head) 缩放。
        - 因果：位置 i 的 query 不能看到 key 位置 j>i；可用上三角为 True 的 bool 与 masked_fill(..., -inf)，
          在最后一维上 softmax 后禁止位置为 0 概率，而非 NaN（注意 -inf 经 softmax 为 0）。
        """
        # ========== 在下方补全；完成后删除本行 raise ==========
        raise NotImplementedError("请补全 CausalSelfAttention.forward（手搓缩放点积 + 因果下三角掩码）")

        # 实现提示（可删，实现后请去掉不会被执行到的死代码/占位注释）：
        # b, t, c = x.size()
        # q, k, v 由 self.w_q / w_k / w_v 得到，再 view 为 (B, n_head, T, d_head)
        # att = (q @ k^T) / sqrt(self.d_head)
        # 构造 (T, T) 的因果上三角（或对 j>i 的掩码）并 masked_fill
        # att = dropout(softmax(att, dim=-1))；y = att @ v
        # y: (B, n_head, T, d_head) → 合并为 (B, T, C) 后经 self.w_o


class FeedForward(nn.Module):
    """
    位置前馈子层：将每个位置的 d_model 维向量先扩到 d_ff，经 GELU 再压回 d_model，并带 Dropout。
    与 Transformer 块中残差、LayerNorm 的配合在 TransformerBlock 里已完成，此处只实现「两路线性 + 非线性」。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_p = float(dropout)
        # ========== 在下方注册子层（如两个 nn.Linear、GELU、nn.Dropout），或 nn.Sequential；补全后删除下一行 ==========
        raise NotImplementedError("请补全 FeedForward.__init__（d_model→d_ff→GELU→d_model，并带 Dropout）")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)，返回 (B, T, d_model)。
        """
        # ========== 在下方补全；补全后删除下一行 ==========
        raise NotImplementedError("请补全 FeedForward.forward")


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class CharGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 256,
        n_head: int = 8,
        n_layer: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, block_size, d_ff, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享（常见技巧，可注释掉）
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.size()
        assert t <= self.block_size, f"长度 {t} 超过 block_size {self.block_size}"
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
