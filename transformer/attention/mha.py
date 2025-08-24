"""
Copyright (c) 2025. All rights reserved.
"""

"""
Multi-head self-attention mechanism for transformer models.
"""

import math
from typing import Optional

import torch
from transformer.attention.sdpa import scaled_dot_product_attention


class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention mechanism with scaled dot-product attention.

    Implements the multi-head attention from "Attention is All You Need".
    Projects inputs to multiple attention heads, computes scaled dot-product attention
    in parallel, and combines the results with a final linear projection.

    Attributes:
        embed_dim (int): Model embedding dimension
        num_heads (int): Number of attention heads
        apply_causal_mask (bool): Whether to apply causal masking
        head_dim (int): Dimension per attention head (embed_dim // num_heads)
        sqrt_d (float): Square root of head dimension for scaling
        q_proj (torch.nn.Linear): Query projection layer
        k_proj (torch.nn.Linear): Key projection layer
        v_proj (torch.nn.Linear): Value projection layer
        out_proj (torch.nn.Linear): Output projection layer
    """

    embed_dim: int
    num_heads: int
    apply_causal_mask: bool
    head_dim: int
    sqrt_d: float
    q_proj: torch.nn.Linear
    k_proj: torch.nn.Linear
    v_proj: torch.nn.Linear
    out_proj: torch.nn.Linear

    def __init__(self, embed_dim: int, num_heads: int, apply_causal_mask: bool) -> None:
        """Initialize multi-head attention module.

        Args:
            embed_dim (int): Model embedding dimension
            num_heads (int): Number of attention heads
            apply_causal_mask (bool): Whether to apply causal masking for autoregressive generation

        Raises:
            AssertionError: If embed_dim is not divisible by num_heads
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.apply_causal_mask = apply_causal_mask
        assert embed_dim % num_heads == 0, "embedding dim should be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        self.sqrt_d = math.sqrt(self.head_dim)

        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, input: torch.Tensor, kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply multi-head attention to input tensor.

        Performs self-attention when kv is None, or cross-attention when kv is provided.
        Reshapes inputs to multi-head format, applies scaled dot-product attention,
        and projects the result back to the original embedding dimension.

        Args:
            input (torch.Tensor): Query input [batch_size, tgt_len, embed_dim]
            kv (Optional[torch.Tensor]): Key/Value input [batch_size, src_len, embed_dim].
                                       If None, uses input for self-attention.

        Returns:
            torch.Tensor: Attention output [batch_size, tgt_len, embed_dim]
        """
        # input_size = (batch_size, tgt_len, embed_dim)
        # kv_size = (batch_size, seq_len, embed_dim)

        # For cross attention, use encoder outputs as KV
        if kv is None:
            kv = input

        q: torch.Tensor = (
            self.q_proj(input)
            .reshape([input.size(0), input.size(1), self.num_heads, self.head_dim])
            .permute(0, 2, 1, 3)
        )
        k: torch.Tensor = (
            self.k_proj(kv).reshape([kv.size(0), kv.size(1), self.num_heads, self.head_dim]).permute(0, 2, 1, 3)
        )
        v: torch.Tensor = (
            self.v_proj(kv).reshape([kv.size(0), kv.size(1), self.num_heads, self.head_dim]).permute(0, 2, 1, 3)
        )

        # Apply mask - used in self-attention only i.e. when KV is None.
        causal_mask: Optional[torch.Tensor] = None
        if self.apply_causal_mask:
            causal_mask = torch.triu(
                torch.ones(input.size(1), input.size(1)), diagonal=1
            )  # strict mask n+1 is set to 1

        # scaled dot-product attention
        attn_out: torch.Tensor = scaled_dot_product_attention(q, k, v, causal_mask)
        attn_out = attn_out.reshape([input.size(0), input.size(1), self.embed_dim])
        return self.out_proj(attn_out)
