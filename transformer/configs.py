"""
Copyright (c) 2025. All rights reserved.
"""

"""
Configuration classes for transformer models.
"""

from dataclasses import dataclass


@dataclass 
class FFNConfig:
    """Configuration for Feed-Forward Network layers.
    
    Attributes:
        embed_dim (int): Embedding dimension
        latent_dim (int): Hidden layer dimension
        use_moe (bool): Whether to use Mixture of Experts
        num_experts (int): Number of experts for MoE
        capacity (int): Capacity factor for MoE
        alpha (float): Load balancing weight for MoE
        topk (int): Number of experts to route to
    """
    embed_dim: int 
    latent_dim: int
    use_moe: bool
    num_experts: int = 0
    capacity: int = 0 
    alpha: float = 0.0 
    topk: int = 1