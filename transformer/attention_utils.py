"""
Copyright (c) 2025. All rights reserved.
"""

"""
Attention mechanism utilities and factory functions.

Provides unified interface for different attention mechanisms:
- MHA (Multi-Head Attention): Standard transformer attention with multiple heads
- MQA (Multi-Query Attention): Single key/value heads with multiple query heads  
- GQA (Group Query Attention): Grouped key/value heads as compromise between MHA/MQA
"""

from typing import Union
from transformer.attention.mha import MultiHeadAttention
from transformer.attention.mqa import MultiQueryAttention
from transformer.attention.gqa import GroupQueryAttention

ATTENTION_TYPE = Union[MultiHeadAttention, MultiQueryAttention, GroupQueryAttention]


def get_attention(
    attention_type: str, 
    embed_dim: int, 
    num_heads: int, 
    num_groups: int, 
    apply_causal_mask: bool, 
    use_kv_cache: bool

) -> ATTENTION_TYPE:
    """Factory function to create attention mechanisms.
    
    Args:
        attention_type (str): Type of attention ("mha", "mqa", "gqa")
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_groups (int): Number of groups for GQA (ignored for MHA/MQA)
        apply_causal_mask (bool): Whether to apply causal masking (only for MHA)
        
    Returns:
        ATTENTION_TYPE: Configured attention mechanism
        
    Raises:
        ValueError: If attention_type is not supported
    """
    if attention_type == "mha": 
        return MultiHeadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            apply_causal_mask=apply_causal_mask,
            use_kv_cache=use_kv_cache,
        )
    elif attention_type == "mqa": 
        return MultiQueryAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            apply_causal_mask=apply_causal_mask, 
            use_kv_cache=use_kv_cache

        )
    elif attention_type == "gqa": 
        return GroupQueryAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_groups=num_groups,
            apply_causal_mask=apply_causal_mask, 
            use_kv_cache=use_kv_cache
        )
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}. Supported types: 'mha', 'mqa', 'gqa'")