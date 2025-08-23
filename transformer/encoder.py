"""
Copyright (c) 2025. All rights reserved.
"""

"""
Transformer encoder model for sequence processing.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.attention import MultiHeadAttention
from transformer.ffn import FFN


class Encoder(torch.nn.Module):
    """Single transformer encoder layer with self-attention and feedforward."""

    attn: MultiHeadAttention
    ffn: FFN
    norm_1: torch.nn.LayerNorm
    norm_2: torch.nn.LayerNorm

    def __init__(
        self, embed_dim: int, num_heads: int, ffn_latent_dim: int, apply_causal_mask: bool
    ) -> None:
        """Initialize transformer encoder layer.

        Args:
            embed_dim (int): Embedding dimension for the model
            num_heads (int): Number of attention heads
            ffn_latent_dim (int): Hidden dimension in feed-forward network
            apply_causal_mask (bool): Whether to apply causal masking in self-attention
        """
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask)
        self.ffn = FFN(embed_dim=embed_dim, latent_dim=ffn_latent_dim)
        self.norm_1 = torch.nn.LayerNorm(embed_dim)
        self.norm_2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and feedforward with residual connections.

        Uses post-layer normalization: applies attention/FFN first, then adds residual
        connection and normalizes. This is the standard transformer architecture.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        x = self.norm_1(x + self.attn(x))  # post-norm
        x = self.norm_2(x + self.ffn(x))  # post-norm
        return x
