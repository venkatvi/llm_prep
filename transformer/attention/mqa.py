"""
Copyright (c) 2025. All rights reserved.
"""

"""
Multi-Query Attention (MQA) - Efficient attention with single key/value heads.
"""

import torch
from typing import Optional
from transformer.attention.sdpa import scaled_dot_product_attention
from transformer.attention.utils import use_cache

MAX_SEQ_LEN = 128


class MultiQueryAttention(torch.nn.Module):
    """Multi-Query Attention with single key/value heads for efficiency.

    Uses multiple query heads but only single key and value heads,
    reducing memory usage during inference while maintaining performance.

    Args:
        embed_dim (int): Embedding dimension, must be divisible by num_heads
        num_heads (int): Number of query attention heads
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        apply_causal_mask: bool,
        use_kv_cache: bool,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Embed dim should be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.apply_causal_mask = apply_causal_mask
        self.use_kv_cache = use_kv_cache
        if self.use_kv_cache:
            self.kv_cache = None

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)  # Multiple query heads
        self.k_proj = torch.nn.Linear(embed_dim, self.head_dim)  # Single key head
        self.v_proj = torch.nn.Linear(embed_dim, self.head_dim)  # Single value head
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(
        self, input: torch.Tensor, kv: Optional[torch.Tensor], expanding_context: bool
    ) -> torch.Tensor:
        """Apply multi-query attention to input sequence.

        Uses single key/value heads that are broadcast across multiple query heads
        for memory efficiency. Routes to cached implementation when use_kv_cache is enabled.

        Args:
            input (torch.Tensor): Query input [batch_size, seq_len, embed_dim]
            kv (Optional[torch.Tensor]): Key/Value input [batch_size, seq_len, head_dim].
                                       If None, uses input for self-attention.
            expanding_context (bool): Cache expansion mode for cached attention.
                                    Ignored when use_kv_cache is False.

        Returns:
            torch.Tensor: Attention output [batch_size, seq_len, embed_dim]
        """
        # Only use KV caching during inference, not during training or validation
        if use_cache(self):
            return self.forward_with_cache(input, kv, expanding_context)

        B, S, D = input.shape

        if kv is None:
            kv = input
        _, _, kvD = kv.shape
        assert kvD == D, "Head dim should match for q and kv"

        # Project to queries (multiple heads), keys and values (single heads)
        q = (
            self.q_proj(input)
            .reshape(B, S, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # [bs, num_heads, seq_len, head_dim]
        k = self.k_proj(kv).unsqueeze(
            1
        )  # [bs, 1, seq_len, head_dim] - single head, will broadcast
        v = self.v_proj(kv).unsqueeze(
            1
        )  # [bs, 1, seq_len, head_dim] - single head, will broadcast

        causal_mask: Optional[torch.Tensor] = None
        if self.apply_causal_mask:
            causal_mask = torch.triu(torch.ones([S, S]), diagonal=1)

        attn_out = scaled_dot_product_attention(q, k, v, causal_mask)

        attn_out = attn_out.reshape([B, S, self.embed_dim])
        return self.out_proj(attn_out)

    def forward_with_cache(
        self, input: torch.Tensor, kv: Optional[torch.Tensor], expanding_context: bool
    ) -> torch.Tensor:
        """Apply multi-query attention with KV caching for autoregressive generation.

        Optimizes inference using multi-query attention with single key/value heads
        that are broadcast across multiple query heads. Caches the single K,V heads
        across generation steps for efficiency.

        Args:
            input (torch.Tensor): Query input [batch_size, seq_len, embed_dim]
            kv (Optional[torch.Tensor]): Key/Value input [batch_size, seq_len, head_dim].
                                       If None, uses input for self-attention.
                                       Note: For MQA, KV input should match single head_dim.
            expanding_context (bool): If True, cache grows with each step (expanding context).
                                    If False, maintains fixed cache size (sliding window).

        Returns:
            torch.Tensor: Attention output [batch_size, output_seq_len, embed_dim]
                         where output_seq_len is seq_len for first call, 1 for subsequent calls.

        Note:
            MQA uses single key/value heads that are broadcast to all query heads,
            providing memory efficiency compared to full multi-head attention.
        """
        B, S, _ = input.shape

        if kv is None:
            kv = input

        kvB, kvS, kvD = kv.shape
        assert kvD == self.embed_dim, "Dimension mismatch between Q Dim and KV Dim"

        if self.kv_cache is None:
            q = (
                self.q_proj(input)
                .reshape([B, S, self.num_heads, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            k = (
                self.k_proj(kv)
                .reshape([kvB, kvS, 1, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v_proj(kv)
                .reshape([kvB, kvS, 1, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            
            # Pre-allocate fixed-size cache
            preset_k = torch.zeros([kvB, 1, MAX_SEQ_LEN, self.head_dim])
            preset_k[:, :, :kvS, :] = k
            preset_v = torch.zeros([kvB, 1, MAX_SEQ_LEN, self.head_dim])
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
            k_new = (
                self.k_proj(kv[:, -1, :])
                .reshape([B, 1, 1, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            v_new = (
                self.v_proj(kv[:, -1, :])
                .reshape([B, 1, 1, self.head_dim])
                .permute(0, 2, 1, 3)
            )
            
            self.kv_cache["key"][:, :, self.kv_cache["cur_pos"], :] = k_new.squeeze(2)
            self.kv_cache["value"][:, :, self.kv_cache["cur_pos"], :] = v_new.squeeze(2)
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
        out = out.reshape([B, Snew, self.embed_dim])
        return self.out_proj(out)
