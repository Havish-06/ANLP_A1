"""
norm.py — Normalization Modules: LayerNorm (C1/C2/C3/C5) + RMSNorm (C4)
==========================================================================

LAYER NORM  (used in C1, C2, C3, C5)
--------------------------------------
Normalizes across ALL d_model features per token:
    x_hat = (x - mean) / sqrt(var + eps)
    out   = gamma * x_hat + beta

RMS NORM  (used in C4)
-----------------------
LLaMA, GPT-NeoX, T5 and most modern LLMs use RMSNorm instead of LayerNorm.

WHY RMSNorm?
    Zhang & Sennrich (2019) showed that the "re-centering" (subtracting the mean)
    in LayerNorm is largely unnecessary — the "re-scaling" (dividing by RMS)
    does most of the work.

    RMSNorm skips the mean subtraction step entirely:
        RMS(x) = sqrt( (1/d) * sum(x_i^2) )
        x_hat  = x / RMS(x)
        out    = gamma * x_hat

    Benefits:
      - ~8% faster than LayerNorm (no mean computation)
      - Fewer parameters (no beta / shift term needed)
      - Empirically similar or better performance
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization. Used in C1, C2, C3, C5.

    Normalizes each token's feature vector to mean=0, std=1,
    then applies learnable scale (gamma) and shift (beta).

    Args:
        d_model (int): Feature dimension.
        eps     (float): Numerical stability constant.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean  = x.mean(dim=-1, keepdim=True)
        var   = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization. Used in C4.

    Skips mean subtraction. Only rescales by the RMS of the features:
        RMS(x) = sqrt( mean(x^2) )
        out    = (x / RMS(x)) * gamma

    Args:
        d_model (int): Feature dimension.
        eps     (float): Numerical stability constant.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d_model))   # no beta — intentional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS: sqrt of mean of squares, per token
        rms   = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_hat = x / rms
        return self.gamma * x_hat
