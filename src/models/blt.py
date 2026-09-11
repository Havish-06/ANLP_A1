"""
blt.py — Simplified Byte Latent Transformer (BLT) - Fixed Patch Width
====================================================================
Uses Conv1d for the encoder and cross-attention for the decoder.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .norm import LayerNorm
from .positional import SinusoidalPE


# Byte vocabulary: 256 byte values + 3 special tokens
BYTE_VOCAB_SIZE = 259 
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258


class ByteLocalEncoder(nn.Module):
    def __init__(self, d_model: int, patch_size: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        self.byte_embedding = nn.Embedding(BYTE_VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        
        # Simple fixed-width grouping using Conv1d
        self.proj = nn.Conv1d(
            d_model, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        byte_ids: (B, L)
        Returns: (B, num_patches, d_model)
        """
        B, L = byte_ids.shape
        
        # Ensure L is perfectly divisible by patch_size by padding if necessary
        remainder = L % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            byte_ids = F.pad(byte_ids, (0, pad_len), value=PAD_ID)
            
        byte_embeds = self.byte_embedding(byte_ids)  # (B, L, d_model)
        
        # Conv1d expects (B, d_model, L)
        byte_embeds = byte_embeds.transpose(1, 2)
        
        patches = self.proj(byte_embeds)  # (B, d_model, num_patches)
        patches = patches.transpose(1, 2)  # (B, num_patches, d_model)
        
        patches = self.norm(patches)
        patches = self.dropout(patches)
        
        return patches


class ByteLocalDecoder(nn.Module):
    def __init__(
        self, d_model: int, patch_size: int = 4, num_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        # Learned byte-position queries within each patch
        self.byte_queries = nn.Parameter(
            torch.randn(1, patch_size, d_model) * 0.02
        )

        # Cross-attention: byte queries attend to patch representation
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # FFN after cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.output_proj = nn.Linear(d_model, BYTE_VOCAB_SIZE)

    def forward(self, patch_repr: torch.Tensor) -> torch.Tensor:
        """
        patch_repr: (B, num_patches, d_model)
        Returns: (B, num_patches * patch_size, BYTE_VOCAB_SIZE)
        """
        B, num_patches, _ = patch_repr.shape
        P = self.patch_size

        # Process each patch independently to save memory
        # Reshape to (B * num_patches, 1, d_model) for keys/values
        patch_kv = patch_repr.reshape(B * num_patches, 1, self.d_model)

        # Expand byte queries: (B * num_patches, P, d_model)
        queries = self.byte_queries.expand(B * num_patches, -1, -1)

        # Cross-attention: each byte query attends to its patch
        normed_q = self.norm1(queries)
        attn_out = self.cross_attn(normed_q, patch_kv, patch_kv)
        x = queries + attn_out

        # FFN
        normed = self.norm2(x)
        x = x + self.ffn(normed)

        # Project to byte vocabulary
        logits = self.output_proj(x)  # (B*num_patches, P, BYTE_VOCAB_SIZE)

        # Reshape back to (B, num_patches * P, BYTE_VOCAB_SIZE)
        logits = logits.view(B, num_patches * P, BYTE_VOCAB_SIZE)
        return logits


class _PatchEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out = self.attn(normed, normed, normed, mask=mask)
        x = x + self.dropout(attn_out)
        normed = self.norm2(x)
        x = x + self.ffn(normed)
        return x


class _PatchDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm3 = LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, encoder_out: torch.Tensor, causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, normed, normed, mask=causal_mask)
        x = x + self.dropout(attn_out)

        normed = self.norm2(x)
        cross_out = self.cross_attn(normed, encoder_out, encoder_out)
        x = x + self.dropout(cross_out)

        normed = self.norm3(x)
        x = x + self.ffn(normed)
        return x


class BLTTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        patch_size: int = 4,
        dropout: float = 0.1,
        max_len: int = 512,
        **kwargs # Ignore d_local and local_heads
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        self.byte_encoder = ByteLocalEncoder(d_model, patch_size, dropout)
        
        # Max length for patches is max_len / patch_size
        patch_max_len = max_len // patch_size + 2 
        
        self.enc_pos = SinusoidalPE(d_model, max_len=patch_max_len, dropout=dropout)
        self.dec_pos = SinusoidalPE(d_model, max_len=patch_max_len, dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            _PatchEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.enc_norm = LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList([
            _PatchDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dec_norm = LayerNorm(d_model)

        self.tgt_byte_encoder = ByteLocalEncoder(d_model, patch_size, dropout)
        self.byte_decoder = ByteLocalDecoder(d_model=d_model, patch_size=patch_size, dropout=dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        src_bytes: torch.Tensor,
        tgt_bytes: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        L_tgt = tgt_bytes.size(1)

        src_patches = self.byte_encoder(src_bytes)
        src_patches = self.enc_pos(src_patches)

        x = src_patches
        for layer in self.encoder_layers:
            x = layer(x, mask=None)  # Simplified: no padding mask
        encoder_out = self.enc_norm(x)

        tgt_patches = self.tgt_byte_encoder(tgt_bytes)
        tgt_patches = self.dec_pos(tgt_patches)

        T = tgt_patches.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=tgt_bytes.device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        y = tgt_patches
        for layer in self.decoder_layers:
            y = layer(y, encoder_out, causal_mask)
        decoder_out = self.dec_norm(y)

        logits = self.byte_decoder(decoder_out)
        
        # Truncate exact length in case of padding
        logits = logits[:, :L_tgt, :]
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src_bytes: torch.Tensor,
        max_len: int = 512,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = src_bytes.size(0)
        device = src_bytes.device
        P = self.patch_size

        src_patches = self.byte_encoder(src_bytes)
        src_patches = self.enc_pos(src_patches)

        x = src_patches
        for layer in self.encoder_layers:
            x = layer(x, mask=None)
        encoder_out = self.enc_norm(x)

        bos_chunk = torch.full((B, P), PAD_ID, dtype=torch.long, device=device)
        bos_chunk[:, 0] = BOS_ID

        all_bytes = [bos_chunk]
        max_patches = (max_len + P - 1) // P

        for step in range(max_patches):
            tgt_bytes_so_far = torch.cat(all_bytes, dim=1)
            
            tgt_patches = self.tgt_byte_encoder(tgt_bytes_so_far)
            tgt_patches = self.dec_pos(tgt_patches)

            T = tgt_patches.size(1)
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            y = tgt_patches
            for layer in self.decoder_layers:
                y = layer(y, encoder_out, causal_mask)
            decoder_out = self.dec_norm(y)

            last_patch = decoder_out[:, -1:, :]
            byte_logits = self.byte_decoder(last_patch)  # (B, 1*P, VOCAB)
            next_bytes = byte_logits.argmax(dim=-1)

            all_bytes.append(next_bytes)

            if (next_bytes == EOS_ID).any(dim=-1).all():
                break

        decoded = torch.cat(all_bytes, dim=1)
        return decoded[:, :max_len]
