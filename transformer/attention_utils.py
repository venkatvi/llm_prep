from typing import Union
from transformer.attention.mha import MultiHeadAttention
from transformer.attention.mqa import MultiQueryAttention
from transformer.attention.gqa import GroupQueryAttention

ATTENTION_TYPE = Union[MultiHeadAttention, MultiQueryAttention, GroupQueryAttention]

def get_attention(attention_type: str, embed_dim: int, num_heads: int, num_groups:int, apply_causal_mask: bool): 
    if attention_type == "mha": 
        return MultiHeadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            apply_causal_mask=apply_causal_mask
        )
    elif attention_type == "mqa": 
        return MultiQueryAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads
        )
    elif attention_type == "gqa": 
        return GroupQueryAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_groups=num_groups
        )