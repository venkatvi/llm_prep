"""
Copyright (c) 2025. All rights reserved.
"""

"""
Attention mechanisms for transformer models.

This module provides three types of attention mechanisms:

1. **Multi-Head Attention (MHA)**: Standard transformer attention with multiple 
   independent query, key, and value heads. Provides maximum representational 
   capacity but highest memory usage.

2. **Multi-Query Attention (MQA)**: Uses multiple query heads but single key and 
   value heads. Reduces memory usage during inference while maintaining performance.

3. **Group Query Attention (GQA)**: Groups query heads to share key/value heads, 
   providing a balance between MHA and MQA in terms of performance and efficiency.

Classes:
    MultiHeadAttention: Standard multi-head attention implementation
    MultiQueryAttention: Memory-efficient single key/value head attention
    GroupQueryAttention: Grouped attention mechanism balancing efficiency and performance
"""

from .mha import MultiHeadAttention
from .mqa import MultiQueryAttention
from .gqa import GroupQueryAttention

__all__ = ["MultiHeadAttention", "MultiQueryAttention", "GroupQueryAttention"]
