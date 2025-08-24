"""
Copyright (c) 2025. All rights reserved.
"""

"""
Group Query Attention (GQA) - Balanced approach between MHA and MQA.
"""

import torch
from transformer.attention.sdpa import scaled_dot_product_attention
from typing import Optional


class GroupQueryAttention(torch.nn.Module):
    """Group Query Attention with grouped key/value heads for efficiency.

    Groups multiple query heads to share key/value heads, providing a balance
    between full multi-head attention and multi-query attention.

    Args:
        embed_dim (int): Embedding dimension, must be divisible by num_heads and num_groups
        num_heads (int): Total number of query heads
        num_groups (int): Number of key/value groups (num_heads must be divisible by num_groups)
    """

    def __init__(self, embed_dim: int, num_heads: int, num_groups: int, apply_causal_mask: bool) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Embed dim should be divisible by num_heads"
        assert num_heads % num_groups == 0, "num_heads should be divisible by num_groups"
        assert num_groups < num_heads, "num_groups should be less than num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_groups  # How many query heads per group
        self.apply_causal_mask = apply_causal_mask

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)  # All query heads
        self.k_proj = torch.nn.Linear(embed_dim, num_groups * self.head_dim)  # Grouped key heads
        self.v_proj = torch.nn.Linear(embed_dim, num_groups * self.head_dim)  # Grouped value heads
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply group query attention to input sequence.

        Args:
            input (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]

        Returns:
            torch.Tensor: Attention output [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = input.shape

        # Project to queries (all heads) and grouped keys/values
        q = (
            self.q_proj(input).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )  # [bs, num_heads, seq_len, head_dim]
        k = (
            self.k_proj(input).reshape(batch_size, seq_len, self.num_groups, self.head_dim).permute(0, 2, 1, 3)
        )  # [bs, num_groups, seq_len, head_dim]
        v = (
            self.v_proj(input).reshape(batch_size, seq_len, self.num_groups, self.head_dim).permute(0, 2, 1, 3)
        )  # [bs, num_groups, seq_len, head_dim]

        # Repeat each group to match number of query heads
        k = k.repeat_interleave(self.group_size, dim=1)  # [bs, num_heads, seq_len, head_dim]
        v = v.repeat_interleave(self.group_size, dim=1)  # [bs, num_heads, seq_len, head_dim]

        causal_mask: Optional[torch.Tensor] = None
        if self.apply_causal_mask:
            causal_mask = torch.triu(torch.ones(input.size(1), input.size(1)), diagonal=1)
        attn_out = scaled_dot_product_attention(q, k, v, causal_mask)

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_out)
