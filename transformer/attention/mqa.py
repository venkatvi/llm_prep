"""
Copyright (c) 2025. All rights reserved.
"""

"""
Multi-Query Attention (MQA) - Efficient attention with single key/value heads.
"""

import torch
from typing import Optional
from transformer.attention.sdpa import scaled_dot_product_attention


class MultiQueryAttention(torch.nn.Module):
    """Multi-Query Attention with single key/value heads for efficiency.

    Uses multiple query heads but only single key and value heads,
    reducing memory usage during inference while maintaining performance.

    Args:
        embed_dim (int): Embedding dimension, must be divisible by num_heads
        num_heads (int): Number of query attention heads
    """

    def __init__(self, embed_dim: int, num_heads: int, apply_causal_mask: bool) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Embed dim should be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.apply_causal_mask = apply_causal_mask

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)  # Multiple query heads
        self.k_proj = torch.nn.Linear(embed_dim, self.head_dim)  # Single key head
        self.v_proj = torch.nn.Linear(embed_dim, self.head_dim)  # Single value head
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply multi-query attention to input sequence.

        Args:
            input (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]

        Returns:
            torch.Tensor: Attention output [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = input.shape

        # Project to queries (multiple heads), keys and values (single heads)
        q = (
            self.q_proj(input).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )  # [bs, num_heads, seq_len, head_dim]
        k = self.k_proj(input).unsqueeze(1)  # [bs, 1, seq_len, head_dim] - single head, will broadcast
        v = self.v_proj(input).unsqueeze(1)  # [bs, 1, seq_len, head_dim] - single head, will broadcast

        causal_mask: Optional[torch.Tensor] = None
        if self.apply_causal_mask:
            causal_mask = torch.triu(torch.ones([input.size(1), input.size(1)]), diagonal=1)

        attn_out = scaled_dot_product_attention(q, k, v, causal_mask)

        attn_out = attn_out.reshape([input.size(0), input.size(1), self.embed_dim])
        return self.out_proj(attn_out)
