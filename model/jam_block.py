# -*- coding: utf-8 -*-
"""
LingJie-Code-v1-Pro  —  JAM Block (Joint Attention-Conv Module)
纯 PyTorch 手写，不使用任何现成 Transformer 库 / 预训练权重。

每个 JAM Block 含三条通路（Pre-norm RMSNorm + 残差 + LayerScale）：
  1. 全局通路: x = x + gamma_attn * RoPE_Causal_GQA_Attention(RMSNorm(x))
  2. 局部通路: x = x + gamma_conv * Depthwise_Causal_Conv1d(RMSNorm(x))
  3. 门控通路: x = x + gamma_ffn  * SwiGLU_FFN(RMSNorm(x))

LayerScale gamma 全部初始化为 1e-4（防梯度爆炸核心）。

支持两种前向模式：
  - 全序列训练模式: forward(x)  严格因果掩码，深度卷积左侧补零保证看不到未来 token
  - 增量推理模式:   forward(x_t, state)  使用 JAM-DualState Cache
      attn_kv_cache:     [batch, n_kv_heads, seq_len, head_dim]   标准 KV Cache，增量拼接
      cnn_norm_history:  [batch, d_model, kernel_size-1]          CNN 滑动窗口
      关键：CNN 层缓存的是 RMSNorm 之后的隐状态（token-wise，历史归一化结果固定不变）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


# ---------------------------------------------------------------------------
# RoPE 旋转位置编码
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, max_pos: int = 8192):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_pos)

    def _build_cache(self, pos: int):
        device = self.inv_freq.device
        t = torch.arange(pos, device=device).float()
        freqs = torch.outer(t, self.inv_freq)              # [pos, head_dim/2]
        self.cos_cached = freqs.cos().to(device)
        self.sin_cached = freqs.sin().to(device)
        self.cached_len = pos

    def forward(self, seq_len: int, offset: int = 0, device=None):
        need = offset + seq_len
        if need > self.cached_len:
            self._build_cache(max(need, self.cached_len * 2))
        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [batch, n_heads, seq_len, head_dim]，cos/sin: [seq_len, head_dim/2]"""
    b, n, t, hd = x.shape
    cos = cos.view(1, 1, t, hd // 2).to(x.dtype)
    sin = sin.view(1, 1, t, hd // 2).to(x.dtype)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    rotated = torch.cat([x2, -x1], dim=-1)
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, -sin], dim=-1)
    return x * cos_full + rotated * sin_full


# ---------------------------------------------------------------------------
# GQA 因果自注意力（8 Q 头 / 2 KV 头）
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, n_kv_heads: int = 2):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.kv_dim = n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        b, n, t, hd = x.shape
        return x.unsqueeze(2).expand(b, n, n_rep, t, hd).reshape(b, n * n_rep, t, hd)

    def forward(self, x: torch.Tensor, state=None) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        pos_offset = state.attn_len() if state is not None else 0
        rope = self.rope
        cos, sin = rope(t, offset=pos_offset, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if state is not None:
            k, v = state.append_attn_kv(k, v)    # 增量拼接历史 KV
            kk = self._repeat_kv(k, self.n_rep)
            vv = self._repeat_kv(v, self.n_rep)
            if t == 1:
                # q 只有当前 token，keys 全为过去+自己，天然因果
                out = F.scaled_dot_product_attention(q, kk, vv, is_causal=False)
            else:
                # 分块 prefill：对扩展 KV 域应用因果掩码
                S = kk.shape[2]
                key_pos = torch.arange(S, device=x.device).view(1, S)
                query_pos = pos_offset + torch.arange(t, device=x.device).view(t, 1)
                mask = key_pos <= query_pos          # [t, S]，True=可见
                out = F.scaled_dot_product_attention(
                    q, kk, vv, attn_mask=mask, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(
                q,
                self._repeat_kv(k, self.n_rep),
                self._repeat_kv(v, self.n_rep),
                is_causal=True,
            )
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# 深度可分离因果卷积（kernel_size=5, 严格因果: 仅左侧补零 4）
# ---------------------------------------------------------------------------
class DepthwiseCausalConv1d(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.left_pad = kernel_size - 1          # 严格因果，只看过去
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              padding=0, groups=d_model, bias=True)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, t, d]，左侧补零保证因果。
        # 实现为与 forward_step 完全一致的逐 tap 累加，保证训练-推理逐元素一致
        b, t, d = x.shape
        h = F.pad(x.transpose(1, 2), (self.left_pad, 0))      # [b, d, t+k-1]
        w = self.conv.weight.squeeze(1)                        # [d, k]
        out = torch.zeros(b, d, t, device=x.device, dtype=torch.float32)
        for j in range(self.kernel_size):
            out = out + (h[:, :, j:j + t].float() * w[:, j].float()
                         .unsqueeze(0).unsqueeze(-1))
        out = out + self.conv.bias.float().unsqueeze(0).unsqueeze(-1)
        return out.transpose(1, 2).to(x.dtype)

    def forward_chunk(self, x_norm: torch.Tensor, history: torch.Tensor):
        """
        分块 prefill。x_norm: [b, t, d]（RMSNorm 之后），t >= 1。
        窗口 = [history | x_norm]，输出与全序列因果卷积逐元素一致。
        返回 (out [b, t, d], new_history [b, d, k-1])
        """
        b, t, d = x_norm.shape
        h = torch.cat([history.float(), x_norm.transpose(1, 2).float()],
                      dim=-1)                      # [b, d, t+k-1]
        w = self.conv.weight.squeeze(1).float()    # [d, k]
        out = torch.zeros(b, d, t, device=x_norm.device, dtype=torch.float32)
        for j in range(self.kernel_size):
            out = out + h[:, :, j:j + t] * w[:, j].unsqueeze(0).unsqueeze(-1)
        out = out + self.conv.bias.float().unsqueeze(0).unsqueeze(-1)
        new_history = x_norm.transpose(1, 2)[:, :, -(self.kernel_size - 1):]
        return out.transpose(1, 2).to(x_norm.dtype), new_history

    def forward_step(self, x_norm: torch.Tensor, history: torch.Tensor):
        """
        增量单步。x_norm: [b, 1, d]（RMSNorm 之后的隐状态）
        history: [b, d, kernel_size-1] 最近 kernel_size-1 个 norm 后状态
        返回 (out [b, 1, d], new_history [b, d, kernel_size-1])
        """
        x_now = x_norm.transpose(1, 2).float()               # [b, d, 1]
        window = torch.cat([history.float(), x_now], dim=-1)  # [b, d, k]
        w = self.conv.weight.squeeze(1).float()               # [d, k]
        out = torch.zeros_like(x_now)
        for j in range(self.kernel_size):
            out = out + window[:, :, j:j + 1] * w[:, j].unsqueeze(0).unsqueeze(-1)
        out = out + self.conv.bias.float().unsqueeze(0).unsqueeze(-1)
        out = out.to(x_norm.dtype)                            # [b, d, 1]
        new_history = torch.cat([history[:, :, 1:], x_norm.transpose(1, 2)],
                                dim=-1)                       # FIFO
        return out.transpose(1, 2), new_history


# ---------------------------------------------------------------------------
# SwiGLU FFN (d_ff = int(d_model * 2.13))
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# JAM-DualState Cache：联合双状态缓存（每层一份）
# ---------------------------------------------------------------------------
class BlockDualState:
    """单个 JAM Block 的双状态容器（KV 预分配，避免逐步 cat 的分配开销）"""

    def __init__(self, batch_size: int, d_model: int, n_kv_heads: int,
                 head_dim: int, kernel_size: int, device,
                 dtype=torch.float32, max_len: int = 1024):
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.max_len = max_len
        self.attn_k = torch.zeros(batch_size, n_kv_heads, max_len, head_dim,
                                  device=device, dtype=dtype)
        self.attn_v = torch.zeros(batch_size, n_kv_heads, max_len, head_dim,
                                  device=device, dtype=dtype)
        self._len = 0
        # CNN 缓存 RMSNorm 之后的隐状态（token-wise，历史归一化结果固定不变）
        self.conv_hist = torch.zeros(batch_size, d_model, kernel_size - 1,
                                     device=device, dtype=dtype)

    def attn_len(self) -> int:
        return self._len

    def append_attn_kv(self, k: torch.Tensor, v: torch.Tensor):
        t = k.shape[2]
        self.attn_k[:, :, self._len:self._len + t] = k
        self.attn_v[:, :, self._len:self._len + t] = v
        self._len += t
        # 返回有效区间的视图（零拷贝）
        return self.attn_k[:, :, :self._len], self.attn_v[:, :, :self._len]

    def set_conv_hist(self, hist: torch.Tensor):
        self.conv_hist = hist


class JAMDualStateCache:
    """整个模型的联合双状态缓存：每个 JAM Block 一份独立双状态"""

    def __init__(self, model, batch_size: int, device, dtype=torch.float32):
        cfg = model.config
        self.states = [
            BlockDualState(batch_size, cfg["d_model"], cfg["n_kv_heads"],
                           cfg["d_model"] // cfg["n_heads"], cfg["kernel_size"],
                           device, dtype,
                           max_len=cfg.get("max_seq_len", 512) * 2)
            for _ in range(cfg["n_layers"])
        ]


# ---------------------------------------------------------------------------
# JAM Block 主体
# ---------------------------------------------------------------------------
class JAMBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, n_kv_heads: int = 2,
                 kernel_size: int = 5, ffn_mult: float = 2.13,
                 rope: RotaryEmbedding = None, layer_scale_init: float = 1e-4):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        d_ff = int(d_model * ffn_mult)

        self.norm_attn = RMSNorm(d_model)
        self.norm_conv = RMSNorm(d_model)
        self.norm_ffn = RMSNorm(d_model)

        self.attn = CausalSelfAttention(d_model, n_heads, n_kv_heads)
        self.attn.rope = rope
        self.conv = DepthwiseCausalConv1d(d_model, kernel_size)
        self.ffn = SwiGLU(d_model, d_ff)

        # LayerScale：防爆核心，初始值严格 1e-4
        self.gamma_attn = nn.Parameter(torch.full((d_model,), layer_scale_init))
        self.gamma_conv = nn.Parameter(torch.full((d_model,), layer_scale_init))
        self.gamma_ffn = nn.Parameter(torch.full((d_model,), layer_scale_init))

    def forward(self, x: torch.Tensor, state=None) -> torch.Tensor:
        b, t, d = x.shape
        pos_offset = state.attn_len() if state is not None else 0

        # ① 全局通路（Attention + RoPE + GQA + 因果掩码）
        h = self.norm_attn(x)
        x = x + self.gamma_attn * self.attn(h, state=state)

        # ② 局部通路（输入是 RMSNorm 之后的隐状态；缓存亦缓存 norm 后状态）
        h = self.norm_conv(x)
        if state is not None:
            if t == 1:
                conv_out, new_hist = self.conv.forward_step(h, state.conv_hist)
            else:
                # 分块 prefill：窗口 = [历史4个norm状态 | 当前块]，与全序列严格一致
                conv_out, new_hist = self.conv.forward_chunk(
                    h, state.conv_hist)
            state.set_conv_hist(new_hist)
            x = x + self.gamma_conv * conv_out
        else:
            x = x + self.gamma_conv * self.conv.forward_full(h)

        # ③ 门控通路（SwiGLU FFN）
        x = x + self.gamma_ffn * self.ffn(self.norm_ffn(x))
        return x
