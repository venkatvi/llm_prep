"""
Copyright (c) 2025. All rights reserved.
"""

"""
Group Query Attention (GQA) - Balanced approach between MHA and MQA.
"""

import torch
from transformer.attention.sdpa import scaled_dot_product_attention
from transformer.attention.utils import use_cache
from typing import Optional

MAX_SEQ_LEN = 128


class GroupQueryAttention(torch.nn.Module):
    """Group Query Attention with grouped key/value heads for efficiency.

    Groups multiple query heads to share key/value heads, providing a balance
    between full multi-head attention and multi-query attention.

    Args:
        embed_dim (int): Embedding dimension, must be divisible by num_heads and num_groups
        num_heads (int): Total number of query heads
        num_groups (int): Number of key/value groups (num_heads must be divisible by num_groups)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_groups: int,
        apply_causal_mask: bool,
        use_kv_cache: bool,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Embed dim should be divisible by num_heads"
        assert (
            num_heads % num_groups == 0
        ), "num_heads should be divisible by num_groups"
        assert num_groups < num_heads, "num_groups should be less than num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_groups  # How many query heads per group
        self.apply_causal_mask = apply_causal_mask
        self.use_kv_cache = use_kv_cache
        self.kv_cache = None

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)  # All query heads
        self.k_proj = torch.nn.Linear(
            embed_dim, num_groups * self.head_dim
        )  # Grouped key heads
        self.v_proj = torch.nn.Linear(
            embed_dim, num_groups * self.head_dim
        )  # Grouped value heads
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(
        self, input: torch.Tensor, kv: Optional[torch.Tensor], expanding_context: bool
    ) -> torch.Tensor:
        """Apply group query attention to input sequence.

        Groups multiple query heads to share key/value heads, balancing performance
        and memory efficiency between full MHA and MQA. Routes to cached implementation
        when use_kv_cache is enabled.

        Args:
            input (torch.Tensor): Query input [batch_size, seq_len, embed_dim]
            kv (Optional[torch.Tensor]): Key/Value input [batch_size, seq_len, embed_dim].
                                       If None, uses input for self-attention.
            expanding_context (bool): Cache expansion mode for cached attention.
                                    Ignored when use_kv_cache is False.

        Returns:
            torch.Tensor: Attention output [batch_size, seq_len, embed_dim]
        """
        # Only use KV caching during inference, not during training or validation
        if use_cache(self):
            return self.forward_with_cache(input, kv, expanding_context)

        B, S, _ = input.shape

        if kv is None:
            kv = input

        kvB, kvS, kvD = kv.shape
        kv_head_dim = kvD // self.num_heads
        assert kv_head_dim == self.head_dim, "Head dim should match for q and kv"

        # Project to queries (all heads) and grouped keys/values
        q = (
            self.q_proj(input)
            .reshape(B, S, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # [bs, num_heads, seq_len, head_dim]
        k = (
            self.k_proj(kv)
            .reshape(kvB, kvS, self.num_groups, kv_head_dim)
            .permute(0, 2, 1, 3)
        )  # [bs, num_groups, seq_len, head_dim]
        v = (
            self.v_proj(kv)
            .reshape(kvB, kvS, self.num_groups, kv_head_dim)
            .permute(0, 2, 1, 3)
        )  # [bs, num_groups, seq_len, head_dim]

        # Repeat each group to match number of query heads
        k = k.repeat_interleave(
            self.group_size, dim=1
        )  # [bs, num_heads, seq_len, head_dim]
        v = v.repeat_interleave(
            self.group_size, dim=1
        )  # [bs, num_heads, seq_len, head_dim]

        causal_mask: Optional[torch.Tensor] = None
        if self.apply_causal_mask:
            causal_mask = torch.triu(torch.ones(S, S), diagonal=1)

        attn_out = scaled_dot_product_attention(q, k, v, causal_mask)

        attn_out = attn_out.reshape(B, S, self.embed_dim)
        return self.out_proj(attn_out)

    def forward_with_cache(
        self, input: torch.Tensor, kv: Optional[torch.Tensor], expanding_context: bool
    ) -> torch.Tensor:
        """Apply group query attention with KV caching for autoregressive generation.

        Optimizes inference using group query attention where multiple query heads
        share grouped key/value heads. Provides a balance between full multi-head
        attention and multi-query attention in terms of performance and memory usage.

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
            GQA groups query heads to share key/value heads (num_heads // num_groups queries
            per group), reducing memory usage while maintaining better performance than MQA.
        """
        B, S, _ = input.size()

        if kv is None:
            kv = input

        kvB, kvS, kvD = kv.shape
        kv_head_dim = kvD // self.num_heads
        assert kv_head_dim == self.head_dim, "Head dim should match for q and kv"

        if self.kv_cache is None:
            q = (
                self.q_proj(input)
                .reshape([B, S, self.num_heads, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            k = (
                self.k_proj(kv)
                .reshape([kvB, kvS, self.num_groups, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v_proj(kv)
                .reshape([kvB, kvS, self.num_groups, self.head_dim])
                .permute(0, 2, 1, 3)
            )

            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

            # Pre-allocate fixed-size cache
            preset_k = torch.zeros([kvB, self.num_heads, MAX_SEQ_LEN, self.head_dim])
            preset_k[:, :, :kvS, :] = k
            preset_v = torch.zeros([kvB, self.num_heads, MAX_SEQ_LEN, self.head_dim])
            preset_v[:, :, :kvS, :] = v
            
            self.kv_cache = {
                "key": preset_k,
                "value": preset_v,
                "cur_pos": kvS
            }
            Snew = S
        else:
            if self.kv_cache["cur_pos"] >= MAX_SEQ_LEN:
                raise ValueError("KV Cache exhausted. Need a bigger cache.")
            
            # Store new K,V at current position
            knew = (
                self.k_proj(kv[:, -1, :])
                .reshape(B, 1, self.num_groups, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            vnew = (
                self.v_proj(kv[:, -1, :])
                .reshape(B, 1, self.num_groups, self.head_dim)
                .permute(0, 2, 1, 3)
            )

            knew = knew.repeat_interleave(self.group_size, dim=1)
            vnew = vnew.repeat_interleave(self.group_size, dim=1)
            
            self.kv_cache["key"][:, :, self.kv_cache["cur_pos"], :] = knew.squeeze(2)
            self.kv_cache["value"][:, :, self.kv_cache["cur_pos"], :] = vnew.squeeze(2)
            self.kv_cache["cur_pos"] += 1
            
            # Get all cached K,V up to current position for expanding context
            cur_pos = self.kv_cache["cur_pos"]
            if expanding_context:
                k = self.kv_cache["key"][:, :, :cur_pos, :]
                v = self.kv_cache["value"][:, :, :cur_pos, :]
            else: 
                # Collect K and V values for last S sequences. 
                start_pos = max(0, cur_pos-S)
                k = self.kv_cache["key"][:, :, start_pos:cur_pos, :]
                v = self.kv_cache["value"][:, :, start_pos:cur_pos, :]
            
            q = (
                self.q_proj(input[:, -1, :])
                .reshape([B, 1, self.num_heads, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            Snew = 1

        # Apply causal mask if needed
        causal_mask: Optional[torch.Tensor] = None
        if self.apply_causal_mask:
            if Snew == 1:
                # For single token generation, no mask needed (can attend to all previous)
                causal_mask = None
            else:
                # For full sequence, apply causal mask
                seq_len = k.size(2)  # Current sequence length
                causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        
        out = scaled_dot_product_attention(q, k, v, causal_mask)
        out = out.reshape([B, Snew, self.embed_dim])  # important
        return self.out_proj(out)
