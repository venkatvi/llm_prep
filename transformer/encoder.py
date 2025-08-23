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
    
    def __init__(self, embed_dim: int, num_heads: int, ffn_latent_dim: int, apply_causal_mask: bool) -> None: 
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask)
        self.ffn = FFN(embed_dim=embed_dim, latent_dim=ffn_latent_dim)
        self.norm_1 = torch.nn.LayerNorm(embed_dim)
        self.norm_2 = torch.nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Apply self-attention and feedforward with residual connections."""
        x = self.norm_1(x + self.attn(x))  # post-norm
        x = self.norm_2(x + self.ffn(x))  # post-norm
        return x 