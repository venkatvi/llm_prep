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

    def __init__(self, embed_dim: int, num_heads: int, apply_causal_mask: bool, use_kv_cache: bool) -> None:
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

    def forward(self, input: torch.Tensor, kv: Optional[torch.Tensor], expanding_context: bool) -> torch.Tensor:
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
        if self.use_kv_cache: 
            return self.forward_with_cache(input, kv, expanding_context)
        
        B, S, _ = input.shape

        if kv is None: 
            kv = input
        _, _, kvD = kv.shape 
        assert(kvD == self.head_dim), "Head dim should match for q and kv"

        # Project to queries (multiple heads), keys and values (single heads)
        q = (
            self.q_proj(input).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )  # [bs, num_heads, seq_len, head_dim]
        k = self.k_proj(kv).unsqueeze(1)  # [bs, 1, seq_len, head_dim] - single head, will broadcast
        v = self.v_proj(kv).unsqueeze(1)  # [bs, 1, seq_len, head_dim] - single head, will broadcast

        causal_mask: Optional[torch.Tensor] = None
        if self.apply_causal_mask:
            causal_mask = torch.triu(torch.ones([S, S]), diagonal=1)

        attn_out = scaled_dot_product_attention(q, k, v, causal_mask)

        attn_out = attn_out.reshape([B, S, self.embed_dim])
        return self.out_proj(attn_out)

    def forward_with_cache(self, input: torch.Tensor, kv: Optional[torch.Tensor], expanding_context: bool) -> torch.Tensor:
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
        assert(kvD == self.head_dim), "Dimension mistmatch between Q Dim and KV Head Dim"
        
        if self.kv_cache is None: 
            q = self.q_proj(input).reshape([B, S, self.num_heads, self.head_dim]).permute(0, 2, 1, 3)
            k = self.k_proj(kv).reshape([kvB, kvS, 1, self.head_dim]).permute(0, 2, 1, 3)
            v = self.v_proj(kv).reshape([kvB, kvS, 1, self.head_dim]).permute(0, 2, 1, 3)
            self.kv_cache = {
                "key": k, 
                "value": v
            }
            Snew = S
        else: 
            q = self.q_proj(input[:, -1, :]).reshape([B, 1, self.num_heads, self.head_dim]).permute(0, 2, 1, 3)
            k_new = self.k_proj(kv[:, -1, :]).reshape([B, 1, 1, self.head_dim]).permute(0, 2, 1, 3)
            v_new = self.v_proj(kv[:, -1, :]).reshape([B, 1, 1, self.head_dim]).permute(0, 2, 1, 3)

            all_k = torch.cat([self.kv_cache["key"], k_new], dim=2)
            all_v = torch.cat([self.kv_cache["value"], v_new], dim=2)

            k = all_k[:, :, -S:, :]
            v = all_v[:, :, -S:, :]
            Snew = 1 

            # Cache new values 
            if expanding_context: 
                self.kv_cache["key"] = all_k 
                self.kv_cache["value"] = all_v
            else: 
                self.kv_cache["key"] = k 
                self.kv_cache["value"] = v 

        out = scaled_dot_product_attention(q, k, v, causal_mask=None)
        out = out.reshape([B, Snew, self.embed_dim])
        return self.out_proj(out)

