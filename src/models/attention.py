"""
attention.py — Attention Mechanisms: MHA (C1/C2/C4/C5) + GQA (C3)
=====================================================================

SCALED DOT-PRODUCT ATTENTION  (shared core)
--------------------------------------------
    Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) · V

MULTI-HEAD ATTENTION — MHA  (C1, C2, C4, C5)
---------------------------------------------
h heads, each with its own Q/K/V projections (all of size d_model//h).
Total K/V projections = h (same as Q heads).

GROUPED-QUERY ATTENTION — GQA  (C3)
--------------------------------------
MOTIVATION: In MHA, every attention head has its own Key and Value matrices.
During inference (autoregressive decoding), we must cache K and V for all
previous positions. With h heads this KV cache grows as O(h × seq_len × d_k).
For large models (GPT-4, LLaMA-70B) this becomes the GPU memory bottleneck.

GQA (Ainslie et al. 2023) groups the h query heads into G groups. Each group
shares ONE set of K/V projections. So instead of h K/V heads we only need G:

    G = 1   → Multi-Query Attention (MQA) — extreme sharing
    G = h   → Standard MHA (no sharing)
    1 < G < h → GQA (the sweet spot used in LLaMA-2, Mistral, Gemma)

EXAMPLE (h=8, G=4):
    Q: 8 heads (1 projection per head)
    K: 4 heads (2 query heads share each K)
    V: 4 heads (2 query heads share each V)

    Memory saved:  (8 - 4) / 8 = 50% KV cache reduction
    Quality loss:  minimal (benchmarks show near-MHA quality)

HOW IT WORKS IN CODE:
    1. Project Q into h heads as normal.
    2. Project K and V into G heads only.
    3. "Expand" K/V by repeating each G-head h//G times to match Q shape.
    4. Run standard scaled dot-product attention.

ROPE INTEGRATION (C2):
    If use_rope=True, RoPE is applied to Q and K after splitting into heads,
    before the dot product. V is NOT rotated.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.positional import RotaryPE


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention.

    Args:
        Q:          (B, heads_q, T_q, d_k)
        K:          (B, heads_k, T_k, d_k)   — heads_k may be < heads_q in GQA
        V:          (B, heads_v, T_k, d_v)   — heads_v == heads_k
        mask:       bool tensor, True = mask out. Broadcastable to (B, heads, T_q, T_k)
        dropout_p:  dropout probability applied to attention weights
        training:   whether the model is in training mode

    Returns:
        (B, heads_q, T_q, d_v)
    """
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, T_q, T_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    # Attention dropout: regularise the attention distribution itself
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=training)
    return torch.matmul(weights, V)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA). Used in C1, C4, C5.
    Optionally applies RoPE to Q/K for C2.

    Args:
        d_model   (int):  Model dimension.
        num_heads (int):  Number of attention heads (h).
        dropout   (float): Dropout on output.
        use_rope  (bool): If True, apply RoPE to Q/K (for C2).
        max_len   (int):  Max sequence length (for RoPE precomputation).
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        dropout:   float = 0.1,
        use_rope:  bool  = False,
        max_len:   int   = 512,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)         # output dropout
        self.attn_dropout_p = dropout              # attention-weight dropout

        # Optional RoPE — applied inside attention to Q and K
        self.rope = RotaryPE(self.d_k, max_len) if use_rope else None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, heads, T, d_k)"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        """(B, heads, T, d_k) → (B, T, d_model)"""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d_k)

    def forward(self, query, key, value, mask=None):
        Q = self._split(self.W_Q(query))   # (B, h, T_q, d_k)
        K = self._split(self.W_K(key))     # (B, h, T_k, d_k)
        V = self._split(self.W_V(value))   # (B, h, T_k, d_k)

        # Apply RoPE to Q and K if enabled (C2 only)
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        out = scaled_dot_product_attention(Q, K, V, mask,
                                           dropout_p=self.attn_dropout_p,
                                           training=self.training)   # (B, h, T_q, d_k)
        return self.dropout(self.W_O(self._merge(out)))


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA). Used in C3.

    h query heads share G key/value head groups (h must be divisible by G).
    Reduces KV cache memory by factor h/G compared to MHA.

    Args:
        d_model    (int):  Model dimension.
        num_heads  (int):  Number of QUERY heads (h).
        num_kv_groups (int): Number of K/V groups (G). Default h//2.
        dropout    (float): Output dropout.
    """

    def __init__(
        self,
        d_model:      int,
        num_heads:    int,
        num_kv_groups: int = None,
        dropout:      float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        if num_kv_groups is None:
            num_kv_groups = max(1, num_heads // 2)    # default: half the Q heads
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads     = num_heads
        self.num_kv_groups = num_kv_groups
        self.heads_per_group = num_heads // num_kv_groups   # how many Q heads share each K/V
        self.d_k = d_model // num_heads

        # Q: full h heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        # K/V: only G groups × d_k each
        self.W_K = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)         # output dropout
        self.attn_dropout_p = dropout              # attention-weight dropout

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        B, T_k, _ = key.shape

        # Project Q into h heads
        Q = self.W_Q(query).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        # (B, h, T_q, d_k)

        # Project K/V into G groups
        K = self.W_K(key).view(B, T_k, self.num_kv_groups, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(B, T_k, self.num_kv_groups, self.d_k).transpose(1, 2)
        # (B, G, T_k, d_k)

        # Expand K/V: each group is repeated heads_per_group times
        # so K/V shape becomes (B, h, T_k, d_k) matching Q
        K = K.repeat_interleave(self.heads_per_group, dim=1)   # (B, h, T_k, d_k)
        V = V.repeat_interleave(self.heads_per_group, dim=1)   # (B, h, T_k, d_k)

        # Standard scaled dot-product attention
        out = scaled_dot_product_attention(Q, K, V, mask,
                                           dropout_p=self.attn_dropout_p,
                                           training=self.training)      # (B, h, T_q, d_k)

        # Merge heads and project
        merged = out.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.d_k)
        return self.dropout(self.W_O(merged))
