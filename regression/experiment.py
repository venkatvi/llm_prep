"""
Copyright (c) 2025. All rights reserved.
"""

"""
Experiment classes for regression and transformer model training.
"""

import os
import sys
from typing import Optional

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import prepare_data
from e_linear_reg import LinearRegressionModel
from e_non_linear_reg import MLP
from h_transformer import (
    ARTransformerModel,
    EncoderDecoderWrapper,
    RegressionTransformerModel,
)

from lib.configs import ExperimentConfig
from lib.experiment import Experiment
from regression.configs import AutoregressiveDecodeConfig
from lib.train import (
    ar_predict,
    predict_encoder_decoder,
    predict_model,
    split_data,
    train_encoder_decoder,
    train_model,
    train_model_with_dataloader,
)
from lib.utils import plot_results


class RegressionExperiment(Experiment):
    """Experiment class for regression model training and evaluation.

    Handles training of linear and non-linear regression models with support for
    both DataLoader-based batch processing and direct tensor operations.

    Attributes:
        model (torch.nn.Module): Regression model (Linear or MLP)
        inputs (torch.Tensor): Generated input data
        targets (torch.Tensor): Generated target data
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize regression experiment with model and synthetic data.

        Args:
            config (ExperimentConfig): Complete experiment configuration
        """
        super().__init__(config)
        self.model: torch.nn.Module = self.define_model()
        # Generate synthetic training data
        self.inputs: torch.Tensor
        self.targets: torch.Tensor
        self.inputs, self.targets = self.model.generate_data(
            self.config.data.fix_random_seed
        )

    def define_model(self) -> torch.nn.Module:
        """Create regression model based on experiment type.

        Returns:
            torch.nn.Module: LinearRegressionModel for 'linear' type,
                           MLP for 'nlinear' or other types
        """
        if self.config.type == "linear":
            return LinearRegressionModel(self.config.model)
        else:
            return MLP(self.config.model)

    def train(self) -> None:
        """Execute training with DataLoader or direct tensor processing.

        Supports two training modes:
        1. DataLoader mode: Creates temporary CSV files and uses batch processing
        2. Direct tensor mode: Processes all data at once

        The mode is determined by config.data.use_dataloader setting.

        Raises:
            RuntimeError: If training fails during DataLoader processing
        """
        # Choose training mode: DataLoader vs direct tensor processing
        if self.config.data.use_dataloader:
            # Split data for DataLoader training
            train_inputs: torch.Tensor
            train_targets: torch.Tensor
            val_inputs: torch.Tensor
            val_targets: torch.Tensor
            train_inputs, train_targets, val_inputs, val_targets = split_data(
                self.inputs, self.targets
            )

            # Create DataLoaders and temporary CSV files
            train_dataloader: DataLoader
            train_dataset_file_name: str
            val_dataloader: DataLoader
            val_dataset_file_name: str
            train_dataloader, train_dataset_file_name = prepare_data(
                train_inputs,
                train_targets,
                suffix="_train.csv",
                batch_size=self.config.data.training_batch_size,
            )
            val_dataloader, val_dataset_file_name = prepare_data(
                val_inputs, val_targets, suffix="_val.csv"
            )

            try:
                # Train with DataLoader (batch processing)
                self.train_loss: float
                self.val_loss: float
                self.train_loss, self.val_loss = train_model_with_dataloader(
                    self.model, self.train_context, train_dataloader, val_dataloader
                )
            except Exception:
                # Clean up temporary files on error
                os.remove(train_dataset_file_name)
                os.remove(val_dataset_file_name)
                raise RuntimeError("Training Failed")

            # Clean up temporary files after successful training
            os.remove(train_dataset_file_name)
            os.remove(val_dataset_file_name)

        else:
            # Train with direct tensor processing (all data at once)
            self.train_loss, self.val_loss = train_model(
                self.model, self.train_context, self.inputs, self.targets
            )

        self.save()

    def predict(self) -> torch.Tensor:
        """Generate predictions and log metrics to TensorBoard.

        Returns:
            torch.Tensor: Model predictions for the input data
        """
        return predict_model(
            self.model, self.inputs, self.targets, self.logs_dir, self.config.name
        )

    def plot_results(self, y_hat: torch.Tensor) -> None:
        """Create scatter plot visualization in TensorBoard.

        Generates a scatter plot comparing ground truth targets with predictions.
        Only creates plot if input and target tensors have matching dimensions.

        Args:
            y_hat (torch.Tensor): Model predictions to visualize
        """
        if self.inputs.dim() == self.targets.dim():
            plot_results(
                self.inputs,
                self.targets,
                y_hat,
                self.train_context.tensorboard_log_dir,
                self.train_context.run_name,
            )


class TransformerExperiment(RegressionExperiment):
    """Experiment class for transformer models with autoregressive capabilities.

    Extends RegressionExperiment to support both regression and autoregressive modes.
    In autoregressive mode, enables token-by-token generation for sequence prediction.
    """

    decode_config: "AutoregressiveDecodeConfig"
    autoregressive: bool

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize transformer experiment.

        Args:
            config (ExperimentConfig): Complete experiment configuration
            autoregressive (bool): Whether to use autoregressive generation mode
        """
        super().__init__(config)
        self.decode_config = config.model.decode_config

    def define_model(self) -> torch.nn.Module:
        """Create appropriate transformer model based on mode.

        Returns:
            ARTransformerModel for autoregressive generation or
            RegressionTransformerModel for scalar prediction
        """
        if self.config.autoregressive:
            return ARTransformerModel(self.config.model)
        else:
            return RegressionTransformerModel(self.config.model)

    def predict_autoregressively(
        self, input: torch.Tensor, num_steps_override: Optional[int] = None
    ) -> torch.Tensor:
        """Generate sequences using autoregressive prediction.

        Args:
            input (torch.Tensor): Initial input sequence
            num_steps_override (Optional[int]): Override for number of generation steps

        Returns:
            torch.Tensor: Generated sequence tokens
        """
        if num_steps_override is not None:
            num_steps = num_steps_override
        else:
            num_steps = self.decode_config.num_steps
        return ar_predict(
            self.model,
            input,
            num_steps,
            self.decode_config.expanding_context,
            self.decode_config.max_seq_len,
            self.train_context.tensorboard_log_dir,
            self.train_context.run_name,
        )


class EncoderDecoderExperiment(RegressionExperiment):
    """Experiment class for transformer models with autoregressive capabilities.

    Extends RegressionExperiment to support both regression and autoregressive modes.
    In autoregressive mode, enables token-by-token generation for sequence prediction.
    """

    decode_config: "AutoregressiveDecodeConfig"

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize encoder-decoder experiment.

        Args:
            config (ExperimentConfig): Complete experiment configuration
        """
        super().__init__(config)
        self.decode_config = config.model.decode_config

    def define_model(self) -> torch.nn.Module:
        """Create encoder-decoder transformer model.

        Returns:
            EncoderDecoderWrapper: Transformer model with encoder-decoder architecture
        """
        return EncoderDecoderWrapper(self.config.model)

    def train(self) -> None:
        """Train the encoder-decoder model on sequence-to-sequence tasks.

        Generates synthetic sequence data and trains the model using
        encoder-decoder specific training procedures.
        """
        source_sequences: torch.Tensor
        target_sequences: torch.Tensor
        source_sequences, target_sequences = self.model.generate_data(random_seed=42)
        self.train_loss, self.val_loss = train_encoder_decoder(
            self.model, source_sequences, target_sequences, self.train_context
        )
        self.save()

    def predict(self) -> torch.Tensor:
        """Generate predictions using encoder-decoder architecture.

        Returns:
            torch.Tensor: Generated sequence predictions
        """
        source_sequences: torch.Tensor
        target_sequences: torch.Tensor
        source_sequences, target_sequences = self.model.generate_data(random_seed=42)
        return predict_encoder_decoder(
            self.model,
            source_sequences,
            target_sequences,
            self.decode_config.num_steps,
            self.decode_config.expanding_context,
            self.train_context.tensorboard_log_dir,
            self.train_context.run_name,
        )
