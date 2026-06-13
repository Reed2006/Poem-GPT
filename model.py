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

from prosody import NUM_RHYMES, NUM_TONES


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank 必须 > 0")
        self.base = base_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base_linear.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_linear.out_features, self.rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B) * self.scaling
        return base_out + lora_out


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
        b, t, c = x.size()
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) *(1.0 / (self.d_head ** 0.5))
        
        mask = torch.tril(torch.ones(t, t, device=x.device)).bool()
        att = att.masked_fill(mask == False, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous()
        y = y.view(b, t, self.d_model)
        return self.w_o(y)



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
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)，返回 (B, T, d_model)。
        """
        return self.net(x)


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
        use_prosody: bool = False,
        num_tones: int = NUM_TONES,
        num_rhymes: int = NUM_RHYMES + 1,
        use_aux_loss: bool = False,
        aux_loss_weight: float = 0.1,
        use_theme: bool = False,
        num_themes: int = 0,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.use_prosody = bool(use_prosody)
        self.num_tones = int(num_tones)
        self.num_rhymes = int(num_rhymes)
        self.use_aux_loss = bool(use_aux_loss)
        self.aux_loss_weight = float(aux_loss_weight)
        self.use_theme = bool(use_theme)
        self.num_themes = int(num_themes)
        self.use_lora = False
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        # ======== Prosody Embeddings: START ========
        # 变更说明:
        # - 仅增加 Tone / Rhyme 两类 embedding。
        # - 不引入韵脚位 embedding，也不在推理阶段做硬约束。
        if self.use_prosody:
            self.tone_emb = nn.Embedding(self.num_tones, d_model)
            self.rhyme_emb = nn.Embedding(self.num_rhymes, d_model)
        # ======== Prosody Embeddings: END ========
        # ======== Theme Embedding: START ========
        # 变更说明:
        # - 为每首诗注入全局主题 id embedding。
        # - 相同字在不同主题下可经由该全局条件学习不同语义偏向。
        if self.use_theme:
            if self.num_themes <= 0:
                raise ValueError("启用 theme embedding 时，num_themes 必须 > 0")
            self.theme_emb = nn.Embedding(self.num_themes, d_model)
        # ======== Theme Embedding: END ========
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, block_size, d_ff, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # ======== Prosody Auxiliary Heads: START ========
        # 变更说明:
        # - 训练阶段额外预测下一字的 Tone / Rhyme。
        # - 用辅助损失推动隐藏表示主动学习格律相关信息。
        if self.use_aux_loss:
            self.tone_head = nn.Linear(d_model, self.num_tones)
            self.rhyme_head = nn.Linear(d_model, self.num_rhymes)
        # ======== Prosody Auxiliary Heads: END ========
        # 权重共享（常见技巧，可注释掉）
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        if use_lora:
            self.enable_lora(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # ======== LoRA Support: START ========
    # 变更说明:
    # - 允许在保持底座权重不动的前提下，为注意力与 FFN 线性层挂接 LoRA。
    # - 主题微调阶段只训练 LoRA 与 theme embedding，避免小数据全量更新带来的灾难性遗忘。
    def _set_named_child(self, parent: nn.Module, child_name: str, child: nn.Module) -> None:
        if child_name.isdigit():
            parent[int(child_name)] = child  # type: ignore[index]
        else:
            setattr(parent, child_name, child)

    def enable_lora(self, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        if self.use_lora:
            return
        target_names = []
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name in {"lm_head", "tone_head", "rhyme_head"}:
                continue
            if ".attn." in name or ".ff.net." in name:
                target_names.append(name)
        for name in target_names:
            parent_name, child_name = name.rsplit(".", 1)
            parent = self.get_submodule(parent_name)
            child = self.get_submodule(name)
            wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            self._set_named_child(parent, child_name, wrapped)
        self.use_lora = True
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_dropout = float(dropout)

    def freeze_base_for_theme_lora(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = False
            if "lora_" in name or "theme_emb" in name:
                param.requires_grad = True
    # ======== LoRA Support: END ========

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
        tone_ids: Optional[torch.Tensor] = None,
        rhyme_ids: Optional[torch.Tensor] = None,
        tone_targets: Optional[torch.Tensor] = None,
        rhyme_targets: Optional[torch.Tensor] = None,
        theme_ids: Optional[torch.Tensor] = None,
    ):
        b, t = idx.size()
        assert t <= self.block_size, f"长度 {t} 超过 block_size {self.block_size}"
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        # ======== Prosody Embeddings: START ========
        # 变更说明:
        # - 若提供对齐的 tone/rhyme 序列，则与 token embedding 直接相加。
        # - 生成时也可按当前上下文动态计算后喂入，但不做候选裁剪。
        if self.use_prosody:
            if tone_ids is not None:
                x = x + self.tone_emb(tone_ids)
            if rhyme_ids is not None:
                x = x + self.rhyme_emb(rhyme_ids)
        # ======== Prosody Embeddings: END ========
        # ======== Theme Embedding: START ========
        # 变更说明:
        # - 主题 id 在整首诗范围内广播到每个位置，与 token/pos/prosody embedding 叠加。
        if self.use_theme and theme_ids is not None:
            x = x + self.theme_emb(theme_ids)
        # ======== Theme Embedding: END ========
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        loss = None
        if targets is not None:
            # ======== Form-Stressed Weighted Loss: START ========
            # 变更说明:
            # - 保留原始交叉熵路径，兼容旧训练逻辑。
            # - 当 train.py 传入逐 token 权重时，改用加权交叉熵，让模型更重视结构标记。
            flat_logits = logits.view(-1, self.vocab_size)
            flat_targets = targets.view(-1)
            per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            if loss_weights is None:
                loss = per_token_loss.mean()
            else:
                flat_weights = loss_weights.view(-1).to(per_token_loss.dtype)
                loss = (per_token_loss * flat_weights).sum() / flat_weights.sum().clamp_min(1e-8)
            # ======== Form-Stressed Weighted Loss: END ========
            # ======== Prosody Auxiliary Loss: START ========
            # 变更说明:
            # - 主损失仍是下一字预测。
            # - 在此基础上叠加 Tone / Rhyme 辅助交叉熵，不改变推理接口。
            if self.use_aux_loss and tone_targets is not None and rhyme_targets is not None:
                tone_logits = self.tone_head(x).view(-1, self.num_tones)
                rhyme_logits = self.rhyme_head(x).view(-1, self.num_rhymes)
                tone_loss = F.cross_entropy(tone_logits, tone_targets.view(-1))
                rhyme_loss = F.cross_entropy(rhyme_logits, rhyme_targets.view(-1))
                loss = loss + self.aux_loss_weight * (tone_loss + rhyme_loss)
            # ======== Prosody Auxiliary Loss: END ========
        return logits, loss
