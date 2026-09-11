"""
transformer.py — Configurable Encoder-Decoder Transformer (C1–C5)
==================================================================

This file contains a single transformer implementation that can be
configured to match any of the 5 experimental setups by passing
the appropriate flags:

    Config  PE             Attention  Norm       Tokenization
    C1      SinusoidalPE   MHA        LayerNorm  Subword (handled in dataset.py)
    C2      RoPE           MHA        LayerNorm  Subword
    C3      SinusoidalPE   GQA        LayerNorm  Subword
    C4      SinusoidalPE   MHA        RMSNorm    Subword
    C5      SinusoidalPE   MHA        LayerNorm  BLT (handled in blt.py)

The factory function `build_model(config_name, ...)` at the bottom
returns the right model for a given config.

PRE-LAYER NORM
--------------
We apply LayerNorm / RMSNorm BEFORE each sub-layer (Pre-LN), not after.
Residual path always stays in the original scale → stable gradients.

    Pre-LN:  out = x + SubLayer(Norm(x))
"""

import torch
import torch.nn as nn

from models.norm       import LayerNorm, RMSNorm
from models.attention  import MultiHeadAttention, GroupedQueryAttention
from models.positional import SinusoidalPE


# ---------------------------------------------------------------------------
# Feed-Forward Network (shared across all configs)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise FFN: d_model → d_ff → d_model.
    Applied independently to each token position after attention.
    Adds non-linear capacity that attention alone cannot provide.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoder Layer (configurable attention + norm)
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """
    Pre-LN Encoder Layer:
        x → Norm → Self-Attn → + residual → Norm → FFN → + residual

    Args:
        attn_cls:  Class for attention (MHA or GQA).
        norm_cls:  Class for normalization (LayerNorm or RMSNorm).
        attn_kwargs: Dict of extra kwargs forwarded to attn_cls (e.g. use_rope).
    """

    def __init__(self, d_model, num_heads, d_ff, dropout, attn_cls, norm_cls, attn_kwargs=None):
        super().__init__()
        attn_kwargs = attn_kwargs or {}
        self.self_attn = attn_cls(d_model, num_heads, dropout=dropout, **attn_kwargs)
        self.ffn       = FeedForward(d_model, d_ff, dropout)
        self.norm1     = norm_cls(d_model)
        self.norm2     = norm_cls(d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention with Pre-LN + residual
        normed = self.norm1(x)
        x = x + self.drop(self.self_attn(normed, normed, normed, src_mask))
        # FFN with Pre-LN + residual
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Decoder Layer (configurable attention + norm)
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    Pre-LN Decoder Layer:
        x → Norm → Masked Self-Attn → + residual
          → Norm → Cross-Attn (enc_out) → + residual
          → Norm → FFN → + residual
    """

    def __init__(self, d_model, num_heads, d_ff, dropout, attn_cls, norm_cls, attn_kwargs=None):
        super().__init__()
        attn_kwargs = attn_kwargs or {}
        self.self_attn  = attn_cls(d_model, num_heads, dropout=dropout, **attn_kwargs)
        self.cross_attn = attn_cls(d_model, num_heads, dropout=dropout, **attn_kwargs)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.norm1      = norm_cls(d_model)
        self.norm2      = norm_cls(d_model)
        self.norm3      = norm_cls(d_model)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        # 1. Masked Self-Attention
        normed = self.norm1(x)
        x = x + self.drop(self.self_attn(normed, normed, normed, tgt_mask))
        # 2. Cross-Attention: query=decoder, key/value=encoder output
        normed = self.norm2(x)
        x = x + self.drop(self.cross_attn(normed, enc_out, enc_out, src_mask))
        # 3. FFN
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


# ---------------------------------------------------------------------------
# Encoder & Decoder stacks
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout, attn_cls, norm_cls, attn_kwargs=None):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, attn_cls, norm_cls, attn_kwargs)
            for _ in range(num_layers)
        ])
        self.norm = norm_cls(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout, attn_cls, norm_cls, attn_kwargs=None):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, attn_cls, norm_cls, attn_kwargs)
            for _ in range(num_layers)
        ])
        self.norm = norm_cls(d_model)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full Seq2Seq Transformer (C1 – C4; C5 uses BLTTransformer in blt.py)
# ---------------------------------------------------------------------------

class Seq2SeqTransformer(nn.Module):
    """
    Configurable Encoder-Decoder Transformer.

    Args:
        src_vocab_size (int): Source (cipher) vocabulary size.
        tgt_vocab_size (int): Target (plaintext) vocabulary size.
        d_model        (int): Model / embedding dimension.
        num_heads      (int): Number of attention heads.
        num_layers     (int): Encoder + Decoder depth (each).
        d_ff           (int): FFN hidden dimension.
        max_len        (int): Max sequence length.
        dropout        (float): Dropout rate.
        pad_idx        (int): Padding token index.
        use_rope       (bool): If True → use RoPE inside MHA (C2).
        use_gqa        (bool): If True → use GQA instead of MHA (C3).
        use_rmsnorm    (bool): If True → use RMSNorm instead of LayerNorm (C4).
        num_kv_groups  (int): GQA key-value groups (only used when use_gqa=True).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:     int   = 256,
        num_heads:   int   = 8,
        num_layers:  int   = 4,
        d_ff:        int   = 1024,
        max_len:     int   = 512,
        dropout:     float = 0.1,
        pad_idx:     int   = 0,
        use_rope:    bool  = False,
        use_gqa:     bool  = False,
        use_rmsnorm: bool  = False,
        num_kv_groups: int = None,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # --- Choose norm class ---
        norm_cls = RMSNorm if use_rmsnorm else LayerNorm

        # --- Choose attention class + kwargs ---
        if use_gqa:
            attn_cls    = GroupedQueryAttention
            attn_kwargs = {'num_kv_groups': num_kv_groups} if num_kv_groups else {}
        else:
            attn_cls    = MultiHeadAttention
            attn_kwargs = {'use_rope': use_rope, 'max_len': max_len}

        # --- Embeddings ---
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.scale = d_model ** 0.5

        # --- Positional Encoding ---
        # RoPE is applied inside attention, so we always add SinusoidalPE to embeddings.
        # For C2, SinusoidalPE is still added but RoPE additionally rotates Q/K inside MHA.
        self.pos_enc = SinusoidalPE(d_model, max_len, dropout)

        # --- Encoder / Decoder stacks ---
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout, attn_cls, norm_cls, attn_kwargs)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout, attn_cls, norm_cls, attn_kwargs)

        # --- Output projection with weight tying ---
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.output_proj.weight = self.tgt_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Explicitly zero out padding embeddings after xavier init
        with torch.no_grad():
            self.src_embedding.weight[self.pad_idx].fill_(0)
            self.tgt_embedding.weight[self.pad_idx].fill_(0)

    def make_src_mask(self, src):
        """Padding mask: (B, 1, 1, T_src)"""
        return (src == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        """Combined causal + padding mask: (B, 1, T_tgt, T_tgt)"""
        B, T = tgt.shape
        causal   = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
        causal   = causal.unsqueeze(0).unsqueeze(0)
        pad_mask = (tgt == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return causal | pad_mask

    def forward(self, src, tgt):
        """
        src: (B, T_src) cipher token IDs
        tgt: (B, T_tgt) shifted target IDs  [<BOS>, w1, ..., wN]
        Returns logits: (B, T_tgt, tgt_vocab_size)
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        src_emb = self.pos_enc(self.src_embedding(src) * self.scale)
        enc_out = self.encoder(src_emb, src_mask)

        tgt_emb = self.pos_enc(self.tgt_embedding(tgt) * self.scale)
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask, src_mask)

        return self.output_proj(dec_out)

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx, eos_idx, max_len=200):
        """Greedy inference — no teacher forcing."""
        self.eval()
        src_mask = self.make_src_mask(src)
        src_emb  = self.pos_enc(self.src_embedding(src) * self.scale)
        enc_out  = self.encoder(src_emb, src_mask)
        tgt      = torch.tensor([[bos_idx]], device=src.device)

        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(tgt)
            tgt_emb  = self.pos_enc(self.tgt_embedding(tgt) * self.scale)
            dec_out  = self.decoder(tgt_emb, enc_out, tgt_mask, src_mask)
            next_tok = self.output_proj(dec_out[:, -1, :]).argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_tok], dim=1)
            if next_tok.item() == eos_idx:
                break

        return tgt[0, 1:]


# ---------------------------------------------------------------------------
# Factory: build the right model for a given config name
# ---------------------------------------------------------------------------

def build_model(
    config_name:    str,
    src_vocab_size: int,
    tgt_vocab_size: int,
    pad_idx:        int,
    **kwargs,
) -> Seq2SeqTransformer:
    """
    Returns a Seq2SeqTransformer configured for the given ablation config.

    Args:
        config_name:    One of 'C1', 'C2', 'C3', 'C4'  (C5 uses BLTTransformer).
        src_vocab_size: Encoder vocabulary size.
        tgt_vocab_size: Decoder vocabulary size.
        pad_idx:        Padding token index.
        **kwargs:       Override any default hyperparameter.

    Raises:
        ValueError: If config_name is not C1–C4.
    """
    defaults = dict(
        d_model=256, num_heads=8, num_layers=4, d_ff=1024,
        max_len=512, dropout=0.1, num_kv_groups=4,
    )
    defaults.update(kwargs)

    config_flags = {
        'C1': dict(use_rope=False, use_gqa=False, use_rmsnorm=False),
        'C2': dict(use_rope=True,  use_gqa=False, use_rmsnorm=False),
        'C3': dict(use_rope=False, use_gqa=True,  use_rmsnorm=False),
        'C4': dict(use_rope=False, use_gqa=False, use_rmsnorm=True),
    }

    if config_name not in config_flags:
        raise ValueError(f"Unknown config '{config_name}'. Expected one of C1–C4 (C5 uses BLTTransformer).")

    flags = config_flags[config_name]
    return Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        **{**defaults, **flags},
    )
