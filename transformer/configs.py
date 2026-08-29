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
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing
        activation (str): Activation function type ("relu", "gelu", "swish")
        adaptive_checkpointing (bool): Whether to use adaptive checkpointing
        memory_threshold_mb (float): Memory threshold for adaptive checkpointing
    """
    embed_dim: int 
    latent_dim: int
    use_moe: bool
    num_experts: int = 0
    capacity: int = 0 
    alpha: float = 0.0 
    topk: int = 1
    use_gradient_checkpointing: bool = False
    activation: str = "relu"
    adaptive_checkpointing: bool = False
    memory_threshold_mb: float = 1000.0


@dataclass
class CheckpointingConfig:
    """Configuration for gradient checkpointing across the model.
    
    Attributes:
        enable_checkpointing (bool): Global checkpointing enable flag
        checkpoint_ffn (bool): Checkpoint FFN layers
        checkpoint_attention (bool): Checkpoint attention layers
        checkpoint_frequency (int): Checkpoint every N layers (0 = no layer-level checkpointing)
        memory_threshold_mb (float): Memory threshold for adaptive checkpointing
        use_pytorch_checkpoint (bool): Use PyTorch's built-in checkpointing
        recomputation_limit (int): Maximum number of recomputations per backward pass
        profile_memory (bool): Enable memory profiling during training
    """
    enable_checkpointing: bool = False
    checkpoint_ffn: bool = True
    checkpoint_attention: bool = False
    checkpoint_frequency: int = 0
    memory_threshold_mb: float = 1000.0
    use_pytorch_checkpoint: bool = False
    recomputation_limit: int = 10
    profile_memory: bool = False