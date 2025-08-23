"""
Copyright (c) 2025. All rights reserved.
"""

"""
Transformer-based regression model wrapper.
"""

import os
import sys
from typing import Optional, Tuple

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import EncoderDecoderConfig, TransformerModelConfig

from lib.utils import set_seed
from transformer.transformer_model import (
    AutoregressiveTransformerModel,
    EncoderDecoder,
    TransformerModel,
)


class RegressionTransformerModel(torch.nn.Module):
    """Regression wrapper for transformer model with synthetic data generation.

    This model uses global average pooling to produce a single output value per sample,
    making it suitable for regression tasks where we need to predict a scalar from a sequence.
    """

    config: TransformerModelConfig
    model: TransformerModel

    def __init__(self, config: TransformerModelConfig) -> None:
        """Initialize regression transformer with configuration.

        Args:
            config (TransformerModelConfig): Model configuration parameters
        """
        super().__init__()
        self.config = config
        self.model = TransformerModel(
            input_dim=config.input_dim,
            embed_dim=config.embed_dim,
            ffn_latent_dim=config.ffn_latent_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            output_dim=config.output_dim,
            apply_causal_mask=config.apply_causal_mask,
            max_seq_len=config.max_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through regression transformer model.

        Args:
            x (torch.Tensor): Input sequences of shape [batch_size, seq_len, input_dim]

        Returns:
            torch.Tensor: Model outputs for regression task
        """
        return self.model(x)

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic sequence data for transformer regression.

        Creates random input sequences where the target is the sum of all elements,
        providing a simple regression task that requires understanding the full sequence.

        Args:
            random_seed: Optional seed for reproducible data generation

        Returns:
            Tuple of (input_sequences, targets) where:
            - input_sequences: [num_samples, seq_len, input_dim]
            - targets: [num_samples] scalar regression targets
        """
        if random_seed:
            set_seed(random_seed)

        # Generate 100 samples with specified sequence length and input dimension
        num_samples: int = 100
        sequence_length: int = 32
        input_dim: int = self.config.input_dim
        x: torch.Tensor = torch.rand([num_samples, sequence_length, input_dim])
        # Target is sum of all sequence elements (regression task)
        y: torch.Tensor = torch.sum(x.reshape([num_samples, sequence_length * input_dim]), dim=1)
        return x, y


class ARTransformerModel(torch.nn.Module):
    """Autoregressive transformer wrapper for sequence-to-sequence prediction.

    This model generates data in an autoregressive format where each input sequence
    is paired with the next token in the sequence, enabling next-token prediction training.
    Unlike the regression model, this preserves sequence structure for generative tasks.
    """

    config: TransformerModelConfig
    model: AutoregressiveTransformerModel

    def __init__(self, config: TransformerModelConfig) -> None:
        """Initialize autoregressive transformer with configuration.

        Args:
            config (TransformerModelConfig): Model configuration parameters
        """
        super().__init__()
        self.config = config
        self.model = AutoregressiveTransformerModel(
            input_dim=config.input_dim,
            embed_dim=config.embed_dim,
            ffn_latent_dim=config.ffn_latent_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            output_dim=config.output_dim,
            apply_causal_mask=config.apply_causal_mask,
            max_seq_len=config.max_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoregressive transformer.

        Args:
            x (torch.Tensor): Input sequences of shape [batch_size, seq_len, input_dim]

        Returns:
            torch.Tensor: Next token predictions of shape [batch_size, seq_len, output_dim]
        """
        return self.model(x)

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic sequence data for autoregressive training.

        Creates input-target pairs where each target is the next token in the sequence,
        enabling the model to learn next-token prediction for autoregressive generation.

        Args:
            random_seed: Optional seed for reproducible data generation

        Returns:
            Tuple of (input_sequences, target_sequences) where:
            - input_sequences: [num_samples, seq_len-1, input_dim] current tokens
            - target_sequences: [num_samples, seq_len-1, input_dim] next tokens
        """
        if random_seed:
            set_seed(random_seed)

        # Generate 101 samples to create 100 input-target pairs after shifting
        num_samples: int = 101
        sequence_length: int = 32
        input_dim: int = self.config.input_dim
        sample_sequences: torch.Tensor = torch.rand([num_samples, sequence_length, input_dim])
        # Create autoregressive pairs: input[:-1] -> target[1:]
        x: torch.Tensor = sample_sequences[:, :-1, :]  # All tokens except last
        y: torch.Tensor = sample_sequences[:, 1:, :]  # All tokens except first
        return x, y


class EncoderDecoderWrapper(torch.nn.Module):
    """Wrapper for encoder-decoder transformer architecture.

    Provides a unified interface for sequence-to-sequence tasks using
    separate encoder and decoder transformer stacks.

    Attributes:
        config (EncoderDecoderConfig): Model configuration
        model (EncoderDecoder): Underlying encoder-decoder model
    """

    def __init__(self, config: EncoderDecoderConfig):
        """Initialize encoder-decoder wrapper with configuration.

        Args:
            config (EncoderDecoderConfig): Model configuration parameters
        """
        super().__init__()
        self.config = config
        self.model = EncoderDecoder(
            input_dim=config.input_dim,
            embed_dim=config.embed_dim,
            ffn_latent_dim=config.ffn_latent_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            output_dim=config.output_dim,
            apply_causal_mask=config.apply_causal_mask,
            max_seq_len=config.max_seq_len,
        )

    def forward(self, source_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder-decoder transformer.

        Args:
            source_sequence (torch.Tensor): Input sequence for encoder [batch_size, src_len, input_dim]
            target_sequence (torch.Tensor): Target sequence for decoder [batch_size, tgt_len, input_dim]

        Returns:
            torch.Tensor: Decoded output sequence [batch_size, tgt_len, output_dim]
        """
        return self.model(source_sequence, target_sequence)

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic sequence data for encoder-decoder training.

        Creates source-target sequence pairs for sequence-to-sequence learning.
        The source and target are temporally shifted versions of the same sequence.

        Args:
            random_seed (Optional[int]): Optional seed for reproducible data generation

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (source_sequences, target_sequences) where:
                - source_sequences: [num_samples, seq_len-1, input_dim] encoder input
                - target_sequences: [num_samples, seq_len-1, input_dim] decoder target
        """
        if random_seed:
            set_seed(random_seed)

        # Generate 101 samples to create 100 input-target pairs after shifting
        num_samples: int = 101
        sequence_length: int = 32
        input_dim: int = self.config.input_dim
        sample_sequences: torch.Tensor = torch.rand([num_samples, sequence_length, input_dim])
        # Create autoregressive pairs: input[:-1] -> target[1:]
        x: torch.Tensor = sample_sequences[:, :-1, :]  # All tokens except last
        y: torch.Tensor = sample_sequences[:, 1:, :]  # All tokens except first
        return x, y
