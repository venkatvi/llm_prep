"""
Copyright (c) 2025. All rights reserved.
"""

"""
Scaled Dot-Product Attention (SDPA) implementation.

Core attention mechanism used by all transformer attention variants (MHA, MQA, GQA).
Implements the fundamental attention computation: Attention(Q,K,V) = softmax(QK^T/√d_k)V
with optional masking support for causal and padding constraints.
"""

import math
from typing import Optional

import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention mechanism.

    Core attention function implementing the scaled dot-product attention from
    "Attention is All You Need" (Vaswani et al., 2017):

    Attention(Q,K,V) = softmax(QK^T / √d_k)V

    The scaling factor √d_k prevents the dot products from growing too large,
    which would push the softmax function into regions with extremely small gradients.

    Algorithm:
    1. Compute attention scores: scores = Q @ K^T / √d_k
    2. Apply optional masking (set masked positions to -∞)
    3. Apply softmax normalization: weights = softmax(scores)
    4. Compute weighted sum of values: output = weights @ V

    Args:
        q (torch.Tensor): Query tensor [batch_size, num_heads, tgt_len, head_dim]
                         Contains the "questions" that determine what to attend to
        k (torch.Tensor): Key tensor [batch_size, num_heads, src_len, head_dim]
                         Contains the "keys" that are matched against queries
        v (torch.Tensor): Value tensor [batch_size, num_heads, src_len, head_dim]
                         Contains the actual information to be aggregated
        mask (Optional[torch.Tensor]): Attention mask [tgt_len, src_len] or [src_len, src_len]
                                     Values of 1 indicate positions to mask (set to -∞)
                                     Used for causal masking or padding masking

    Returns:
        torch.Tensor: Attention output [batch_size, tgt_len, num_heads, head_dim]
                     Contextualized representations combining information from all positions

    Note:
        - For self-attention: Q, K, V all come from the same input sequence
        - For cross-attention: Q comes from target, K and V from source sequence
        - Causal masking prevents attention to future positions (autoregressive)
        - The scaling factor √d_k is crucial for training stability

    Example:
        >>> q = torch.randn(2, 8, 10, 64)  # [batch=2, heads=8, seq=10, dim=64]
        >>> k = torch.randn(2, 8, 10, 64)
        >>> v = torch.randn(2, 8, 10, 64)
        >>> output = scaled_dot_product_attention(q, k, v)  # [2, 10, 8, 64]

        >>> # With causal mask for autoregressive attention
        >>> causal_mask = torch.triu(torch.ones(10, 10), diagonal=1)
        >>> output = scaled_dot_product_attention(q, k, v, causal_mask)
    """
    sqrt_d: float = math.sqrt(q.size(-1))

    # Inbuilt torch ops
    # https://docs.pytorch.org/docs/stable/generated/torch.baddbmm.html
    # https://docs.pytorch.org/docs/stable/generated/torch.bmm.html

    # 1, 1, 3, 32 @ 1, 1, 32, 3 ==> 1, 1, 3, 3
    scores: torch.Tensor = q @ k.transpose(
        -2, -1
    )  # bs, n_heads, tgt_len, head_dim @ bs, n_heads, head_dim, seq_len
    scores = scores / sqrt_d  # bs, n_heads, tgt_len, seq_len

    if mask is not None:
        scores = scores.masked_fill(
            mask.bool(), float("-inf")
        )  # set all 1s to -inf, so that e^-inf = 0

    scores = torch.softmax(scores, dim=-1)  # compute softmax scores along the last dim

    # 1, 1, 3, 3 @ 1, 1, 3, 32 --> 1, 1,3, 32
    attn_out: torch.Tensor = (
        scores @ v
    )  # bs, n_heads, tgt_len, seq_len @ bs, n_heads, seq_len, head_dim --> bs, n_heads, tgt_len, head_dim
    attn_out = attn_out.transpose(1, 2)  # bs, tgt_len, n_heads, head_dim
    return attn_out
