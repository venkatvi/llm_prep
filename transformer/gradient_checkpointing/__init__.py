"""
Copyright (c) 2025. All rights reserved.
"""

"""
Gradient checkpointing module for memory-efficient transformer training.

This module provides custom autograd functions and utilities for implementing
gradient checkpointing in transformer models, trading compute for memory efficiency.
"""

from .ffn import (
    CheckpointedFFN,
    CheckpointedFFNFunction,
    AdaptiveCheckpointedFFN,
    memory_profiler,
    compare_ffn_implementations
)

__all__ = [
    "CheckpointedFFN",
    "CheckpointedFFNFunction", 
    "AdaptiveCheckpointedFFN",
    "memory_profiler",
    "compare_ffn_implementations"
]