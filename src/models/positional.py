"""
positional.py — Positional Encodings: Sinusoidal (C1/C3/C4/C5) + RoPE (C2)
=============================================================================

SINUSOIDAL PE  (used in C1, C3, C4, C5)
----------------------------------------
Fixed, additive encoding. Adds a precomputed sin/cos pattern to every
token embedding before the first attention layer.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

ROTARY POSITION EMBEDDING — RoPE  (used in C2)
-----------------------------------------------
Instead of adding position info to the embedding, RoPE *rotates* the Q and K
vectors in a position-dependent way INSIDE each attention head, right before
the dot product. This has two key advantages:

  1. RELATIVE POSITIONS: The dot product Q_i · K_j ends up depending only on
     the *relative* position (i - j), not absolute positions. This lets the
     model generalize better to sequence lengths unseen during training.

  2. NO EXTRA PARAMETERS: RoPE is a deterministic rotation — nothing to learn,
     no added parameters.

  3. USED IN PRACTICE: LLaMA, GPT-NeoX, Falcon all use RoPE. It's now the
     dominant positional encoding in LLMs.

HOW ROPE WORKS (the math, simplified):
    Each d_k-dimensional Q/K vector is split into d_k/2 pairs (x_{2i}, x_{2i+1}).
    Each pair is rotated by angle θ_i × pos, where θ_i = 10000^(-2i/d_k):

        [x_{2i},   x_{2i+1}] · Rotation(θ_i × pos)
        = [x_{2i}·cos - x_{2i+1}·sin,  x_{2i}·sin + x_{2i+1}·cos]

    Because both Q and K are rotated, their dot product:
        (R_m · q) · (R_n · k) = q · (R_{n-m} · k)
    depends only on relative position (n - m). Magic!
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal PE (C1, C3, C4, C5)
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """
    Sinusoidal Positional Encoding — additive, fixed, no learnable params.

    Args:
        d_model (int): Embedding dimension.
        max_len (int): Max sequence length to precompute PE for.
        dropout (float): Dropout after adding PE.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)  →  x + PE[:, :T, :]"""
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ---------------------------------------------------------------------------
# RoPE (C2) — applied inside attention, NOT to embeddings
# ---------------------------------------------------------------------------

class RotaryPE(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Unlike SinusoidalPE, RoPE is applied directly to Q and K vectors
    inside the MultiHeadAttention block, per head, before the dot product.

    It does NOT modify V, and it is NOT added to token embeddings.

    Args:
        d_k     (int): Dimension per attention head (d_model // num_heads).
        max_len (int): Max sequence length to precompute rotations for.
    """

    def __init__(self, d_k: int, max_len: int = 512):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        # Precompute sin and cos tables for all positions and frequency pairs
        # θ_i = 10000^(-2i / d_k),  i = 0..d_k/2-1
        theta = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))   # (d_k/2,)
        positions = torch.arange(max_len).float()                             # (max_len,)
        freqs = torch.outer(positions, theta)                                 # (max_len, d_k/2)

        # We'll need sin and cos of each frequency at each position
        self.register_buffer('cos_table', freqs.cos())   # (max_len, d_k/2)
        self.register_buffer('sin_table', freqs.sin())   # (max_len, d_k/2)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given x of shape (..., d_k), split into even/odd halves and rotate:
            [..., x0, x1, x2, x3] → [..., -x1, x0, -x3, x2]
        This is the "rotation by 90 degrees" partner needed for the cos/sin formula.
        """
        x1 = x[..., 0::2]   # even indices  (..., d_k/2)
        x2 = x[..., 1::2]   # odd indices   (..., d_k/2)
        # Stack as (-x2, x1) then interleave back into (..., d_k)
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE rotation to a Q or K tensor.

        Args:
            x: (B, heads, T, d_k)

        Returns:
            Rotated tensor of the same shape.
        """
        T = x.size(2)
        cos = self.cos_table[:T, :]   # (T, d_k/2)
        sin = self.sin_table[:T, :]   # (T, d_k/2)

        # Expand cos/sin to match x's shape for broadcasting:
        #   (T, d_k/2) → (1, 1, T, d_k/2) → repeat to full d_k
        cos = torch.repeat_interleave(cos, 2, dim=-1)   # (T, d_k)
        sin = torch.repeat_interleave(sin, 2, dim=-1)   # (T, d_k)
        cos = cos.unsqueeze(0).unsqueeze(0)              # (1, 1, T, d_k)
        sin = sin.unsqueeze(0).unsqueeze(0)              # (1, 1, T, d_k)

        # Rotation formula: x·cos + rotate_half(x)·sin
        return x * cos + self._rotate_half(x) * sin
