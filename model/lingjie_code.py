# -*- coding: utf-8 -*-
"""
LingJie-Code-v1-Pro  —  完整模型
从 torch.nn.init 随机初始化开始，纯 PyTorch 手写 JAM 架构堆叠。
不加载任何开源预训练权重，不使用 transformers.AutoModel / nn.TransformerEncoderLayer。

主干：Embedding -> N x JAMBlock -> RMSNorm -> LM Head
辅助：5 个级联决策 Head（见 heads.py）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from jam_block import (JAMBlock, JAMDualStateCache, RMSNorm, RotaryEmbedding)
from heads import MultiHeads


DEFAULT_CONFIG = {
    "vocab_size": 9600,
    "d_model": 640,
    "n_layers": 52,
    "n_heads": 8,           # 8 个 Q 头
    "n_kv_heads": 2,        # GQA：2 个 KV 头
    "kernel_size": 5,
    "ffn_mult": 2.13,
    "max_seq_len": 512,
    "dropout": 0.0,
}


class LingJieCodeModel(nn.Module):
    def __init__(self, config: dict = None, pad_token_id: int = 0,
                 resp_token_id: int = 4):
        super().__init__()
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        cfg = self.config

        self.pad_token_id = pad_token_id
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.rope = RotaryEmbedding(cfg["d_model"] // cfg["n_heads"],
                                    max_pos=cfg["max_seq_len"] * 4)
        self.blocks = nn.ModuleList([
            JAMBlock(cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"],
                     cfg["kernel_size"], cfg["ffn_mult"], self.rope)
            for _ in range(cfg["n_layers"])
        ])
        self.norm_f = RMSNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.heads = MultiHeads(cfg["d_model"], pad_token_id, resp_token_id)

        # 从零随机初始化（绝对不加载任何预训练权重）
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                state=None, pool_idx: torch.Tensor = None, with_heads: bool = True):
        """
        input_ids: [b, t]
        attention_mask: [b, t]，1=有效 / 0=padding；None 视为全 1
        pool_idx: 各样本 <resp> 决策点位置；None 时自动计算
        with_heads: 增量解码的后续步可关闭 Head 计算以降低每步开销
        返回 dict: lm_logits [b, t, vocab] + 各 head logits
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = self.tok_emb(input_ids)
        for i, block in enumerate(self.blocks):
            blk_state = state.states[i] if state is not None else None
            x = block(x, state=blk_state)
        x = self.norm_f(x)
        lm_logits = self.lm_head(x)
        if not with_heads:
            return {"lm_logits": lm_logits}
        if pool_idx is None:
            pool_idx = self.heads.pool.find_pool_idx(
                input_ids, self.heads.pool.resp_token_id, self.pad_token_id)
        head_logits = self.heads(x, pool_idx)
        return {"lm_logits": lm_logits, **head_logits}

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 96,
                 temperature: float = 0.7, top_k: int = 40,
                 eos_token_id: int = 2, pad_token_id: int = 0,
                 use_cache: bool = True, seed: int = None):
        """
        自回归生成（JAM-DualState Cache 增量推理）。
        input_ids: [b, t]，返回 (generated_ids [b, t+new], cache)
        """
        device = input_ids.device
        bsz = input_ids.shape[0]
        if seed is not None:
            torch.manual_seed(seed)

        # 1) 预填充
        cache = JAMDualStateCache(self, bsz, device) if use_cache else None
        state = cache if use_cache else None
        out = self(input_ids, state=state)
        next_logits = out["lm_logits"][:, -1, :]
        generated = input_ids
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = next_logits.float()
            if temperature <= 0:
                next_tok = logits.argmax(dim=-1)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(1)
            next_tok = torch.where(finished, torch.full_like(next_tok, pad_token_id),
                                   next_tok)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_token_id)
            if bool(finished.all()):
                break
            out = self(next_tok.unsqueeze(1), state=state, with_heads=False)
            next_logits = out["lm_logits"][:, -1, :]
        return generated, cache
