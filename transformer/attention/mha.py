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
from transformer.attention.utils import use_cache


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

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        apply_causal_mask: bool,
        use_kv_cache: bool,
    ) -> None:
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
        assert (
            embed_dim % num_heads == 0
        ), "embedding dim should be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        self.sqrt_d = math.sqrt(self.head_dim)

        self.use_kv_cache = use_kv_cache
        self.kv_cache = None

        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, input: torch.Tensor, kv: Optional[torch.Tensor], expanding_context: bool
    ) -> torch.Tensor:
        """Apply multi-head attention to input tensor.

        Performs self-attention when kv is None, or cross-attention when kv is provided.
        Routes to cached implementation when use_kv_cache is enabled, otherwise performs
        standard attention computation.

        Args:
            input (torch.Tensor): Query input [batch_size, tgt_len, embed_dim]
            kv (Optional[torch.Tensor]): Key/Value input [batch_size, src_len, embed_dim].
                                       If None, uses input for self-attention.
            expanding_context (bool): Cache expansion mode for cached attention.
                                    Ignored when use_kv_cache is False.

        Returns:
            torch.Tensor: Attention output [batch_size, tgt_len, embed_dim]
        """
        # Only use KV caching during true inference (not during training loops)
        # Use caching only when explicitly in eval mode AND gradients are disabled
        # AND we're not in a validation context (which uses no_grad but is still part of training)

        if use_cache(self):
            return self.forward_with_cache(input, kv, expanding_context)

        B, S, _ = input.shape

        # For cross attention, use encoder outputs as KV
        if kv is None:
            kv = input

        kvB, kvS, kvD = kv.shape
        assert (
            kvD % self.num_heads == 0
        ), "KV Embed Dim should be divisible by num_heads"
        kv_head_dim = kvD // self.num_heads
        assert kv_head_dim == self.head_dim, "Head dim should match for q and kv"

        q: torch.Tensor = (
            self.q_proj(input)
            .reshape([B, S, self.num_heads, self.head_dim])
            .permute(0, 2, 1, 3)
        )
        k: torch.Tensor = (
            self.k_proj(kv)
            .reshape([kvB, kvS, self.num_heads, kv_head_dim])
            .permute(0, 2, 1, 3)
        )
        v: torch.Tensor = (
            self.v_proj(kv)
            .reshape([kvB, kvS, self.num_heads, kv_head_dim])
            .permute(0, 2, 1, 3)
        )

        # Apply mask - used in self-attention only i.e. when KV is None.
        causal_mask: Optional[torch.Tensor] = None
        if self.apply_causal_mask:
            causal_mask = torch.triu(
                torch.ones(S, S), diagonal=1
            )  # strict mask n+1 is set to 1

        # scaled dot-product attention
        attn_out: torch.Tensor = scaled_dot_product_attention(q, k, v, causal_mask)
        attn_out = attn_out.reshape([B, S, self.embed_dim])
        return self.out_proj(attn_out)

    def forward_with_cache(
        self, input: torch.Tensor, kv: Optional[torch.Tensor], expanding_context: bool
    ) -> torch.Tensor:
        """Apply multi-head attention with KV caching for autoregressive generation.

        Optimizes inference by caching key and value tensors across generation steps,
        avoiding recomputation of previous tokens. Supports both fixed-size sliding
        window caching and expanding context caching.

        Args:
            input (torch.Tensor): Query input [batch_size, seq_len, embed_dim]
            kv (Optional[torch.Tensor]): Key/Value input [batch_size, seq_len, embed_dim].
                                       If None, uses input for self-attention.
            expanding_context (bool): If True, cache grows with each step (expanding context).
                                    If False, maintains fixed cache size (sliding window).

        Returns:
            torch.Tensor: Attention output [batch_size, output_seq_len, embed_dim]
                         where output_seq_len is seq_len for first call, 1 for subsequent calls.

        Note:
            For autoregressive generation, input typically contains only the last token
            (seq_len=1) except for the initial call which processes the full sequence.
        """
        B, S, _ = input.size()

        if kv is None:
            kv = input

        kvB, kvS, kvD = kv.size()
        assert kvD % self.num_heads == 0, "KV Embed Dim should be divislbe by num_heads"
        kv_head_dim = kvD // self.num_heads
        assert kv_head_dim == self.head_dim, "Head dim should match for q and kv"

        # 1. Cache is none
        # First call - Self attention inputs only
        if self.kv_cache is None:
            # training will always hit this.
            q = (
                self.q_proj(input)
                .reshape([B, S, self.num_heads, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            k = (
                self.k_proj(kv)
                .reshape([kvB, kvS, self.num_heads, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v_proj(kv)
                .reshape([kvB, kvS, self.num_heads, self.head_dim])
                .permute(0, 2, 1, 3)
            )

            self.kv_cache = {
                "key": k,
                "value": v,
            }
            Snew = S
        else:
            # Compute projections for last entry of the input
            k_new: torch.Tensor = (
                self.k_proj(kv[:, -1, :])
                .reshape([kvB, 1, self.num_heads, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            v_new: torch.Tensor = (
                self.v_proj(kv[:, -1, :])
                .reshape([kvB, 1, self.num_heads, self.head_dim])
                .permute(0, 2, 1, 3)
            )

            # k_new, v_new [kvB, num_heads, kvS, head_dim]
            all_k: torch.Tensor = torch.cat(
                [self.kv_cache["key"][:, :, :, :], k_new], dim=2
            )
            all_v: torch.Tensor = torch.cat(
                [self.kv_cache["value"][:, :, :, :], v_new], dim=2
            )

            k = all_k[:, :, -S:, :]
            v = all_v[:, :, -S:, :]

            if not expanding_context:
                self.kv_cache["key"] = k
                self.kv_cache["value"] = v
            else:
                self.kv_cache["key"] = all_k
                self.kv_cache["value"] = all_v

            q = (
                self.q_proj(input[:, -1, :])
                .reshape([B, 1, self.num_heads, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            Snew = 1

        # compute attention over last row of Q
        out: torch.Tensor = scaled_dot_product_attention(q, k, v, mask=None)
        out = out.reshape([B, Snew, self.embed_dim])
        return self.out_proj(out)
