"""
Copyright (c) 2025. All rights reserved.
"""

"""
Post-training quantization module for transformer models.

This module provides comprehensive quantization capabilities including:
- INT8 and INT4 quantization schemes
- Calibration dataset processing
- Layer-specific quantization strategies
- Accuracy evaluation and metrics
"""

from .quantizer import (
    TransformerQuantizer,
    QuantizationConfig,
    QuantizationScheme,
    LayerQuantizationStrategy
)

from .calibration import (
    CalibrationDataset,
    CalibrationManager,
    StatisticsCollector
)

from .layers import (
    QuantizedLinear,
    QuantizedEmbedding,
    QuantizedAttention,
    QuantizedFFN
)

from .metrics import (
    QuantizationMetrics,
    AccuracyEvaluator,
    PerplexityEvaluator
)

__all__ = [
    "TransformerQuantizer",
    "QuantizationConfig", 
    "QuantizationScheme",
    "LayerQuantizationStrategy",
    "CalibrationDataset",
    "CalibrationManager",
    "StatisticsCollector",
    "QuantizedLinear",
    "QuantizedEmbedding", 
    "QuantizedAttention",
    "QuantizedFFN",
    "QuantizationMetrics",
    "AccuracyEvaluator",
    "PerplexityEvaluator"
]