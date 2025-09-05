"""
Copyright (c) 2025. All rights reserved.
"""

"""
Automatic Mixed Precision (AMP) module for transformer training.

This module provides advanced gradient scaling and mixed precision training
utilities optimized for transformer models.
"""

from .scaler import (
    GradientScaler,
    ScalerConfig,
    ScalingStrategy,
    AMPTrainingLoop,
    create_optimized_scaler_config
)

__all__ = [
    "GradientScaler",
    "ScalerConfig", 
    "ScalingStrategy",
    "AMPTrainingLoop",
    "create_optimized_scaler_config"
]