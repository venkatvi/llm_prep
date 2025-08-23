"""
Copyright (c) 2025. All rights reserved.
"""

"""
Experiment classes for regression and transformer model training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from dataset import prepare_data
from e_linear_reg import LinearRegressionModel
from e_non_linear_reg import MLP
from h_transformer import RegressionTransformerModel, ARTransformerModel
from lib.experiment import Experiment
from lib.configs import ExperimentConfig
from lib.train import train_model, predict_model, train_model_with_dataloader, split_data
from lib.utils import plot_results

class RegressionExperiment(Experiment):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self.model: torch.nn.Module = self.define_model()
        # Generate synthetic training data
        self.inputs: torch.Tensor
        self.targets: torch.Tensor
        self.inputs, self.targets = self.model.generate_data(self.config.data.fix_random_seed)
        
    
    def define_model(self) -> torch.nn.Module:
        """Create regression model based on experiment type."""
        if self.config.type == "linear":
            return LinearRegressionModel(self.config.model)
        else:
            return MLP(self.config.model)
    
    def train(self) -> None:
        """Execute training with DataLoader or direct tensor processing."""
        # Choose training mode: DataLoader vs direct tensor processing
        if self.config.data.use_dataloader: 
            # Split data for DataLoader training
            train_inputs: torch.Tensor
            train_targets: torch.Tensor
            val_inputs: torch.Tensor
            val_targets: torch.Tensor
            train_inputs, train_targets, val_inputs, val_targets = split_data(self.inputs, self.targets)
            
            # Create DataLoaders and temporary CSV files
            train_dataloader: DataLoader
            train_dataset_file_name: str
            val_dataloader: DataLoader
            val_dataset_file_name: str
            train_dataloader, train_dataset_file_name = prepare_data(train_inputs, train_targets, suffix="_train.csv", batch_size=self.config.data.training_batch_size)
            val_dataloader, val_dataset_file_name = prepare_data(val_inputs, val_targets, suffix="_val.csv")
            
            try:
                # Train with DataLoader (batch processing)
                self.train_loss: float
                self.val_loss: float
                self.train_loss, self.val_loss = train_model_with_dataloader(
                    self.model,
                    self.train_context,
                    train_dataloader,
                    val_dataloader
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
                self.model,
                self.train_context,
                self.inputs,
                self.targets
            )
        
        self.save()

    def predict(self) -> torch.Tensor:
        """Generate predictions and log metrics to TensorBoard."""
        return predict_model(
            self.model,
            self.inputs,
            self.targets,
            self.logs_dir,
            self.config.name
        )
    def plot_results(self, y_hat: torch.Tensor) -> None:
        """Create scatter plot visualization in TensorBoard."""
        if self.inputs.dim() == self.targets.dim():
            plot_results(
                self.inputs,
                self.targets,
                y_hat,
                self.train_context.tensorboard_log_dir,
                self.train_context.run_name)
            
class TransformerExperiment(RegressionExperiment):
    """Experiment class for transformer models with autoregressive capabilities.
    
    Extends RegressionExperiment to support both regression and autoregressive modes.
    In autoregressive mode, enables token-by-token generation for sequence prediction.
    """
    decode_config: 'AutoregressiveDecodeConfig'
    autoregressive: bool
    
    def __init__(self, config: ExperimentConfig, autoregressive: bool) -> None: 
        super().__init__(config)
        self.decode_config = config.model.decode_config
        self.autoregressive = autoregressive
    
    def define_model(self) -> torch.nn.Module: 
        """Create appropriate transformer model based on mode.
        
        Returns:
            ARTransformerModel for autoregressive generation or 
            RegressionTransformerModel for scalar prediction
        """
        if self.autoregressive:
            return ARTransformerModel(self.config.model)
        else:
            return RegressionTransformerModel(self.config.model)
    
    def predict_autoregressively(self, input: torch.Tensor, num_steps_override: Optional[int] = None) -> torch.Tensor: 
        """Generate tokens autoregressively from initial input sequence.
        
        Performs token-by-token generation using the trained model, with support for
        both expanding context (keeping all previous tokens) and sliding window modes.
        
        Args:
            input: Initial sequence [batch_size, seq_len, input_dim]
            num_steps_override: Override number of generation steps
            
        Returns:
            Generated tokens [batch_size, num_generated_tokens, output_dim]
        """
        generation_tokens: list = []
        if num_steps_override is None:
            num_steps_override = self.decode_config.num_steps

        current_input: torch.Tensor = input.clone()
        with torch.no_grad():
            for step in range(num_steps_override): 
                print(f"Generating next_token in {step}, current_input size: {current_input.size()}")
                
                # Generate next token using the model's dedicated method
                next_token: torch.Tensor = self.model.generate_next_token(current_input)
                generation_tokens.append(next_token)
                
                # Update context based on decode configuration
                if self.decode_config.expanding_context: 
                    # Keep expanding context (append new token)
                    current_input = torch.cat([
                        current_input, 
                        next_token, 
                    ], dim=1)
                    # Truncate if exceeds maximum sequence length
                    if current_input.size(1) > self.decode_config.max_seq_len:
                        current_input = current_input[:, -self.decode_config.max_seq_len:, :]
                else:
                    # Sliding window approach (remove first, add new)
                    current_input = torch.cat([
                        current_input[:, 1:, :],  # Remove first token
                        next_token,               # Add generated token
                    ], dim=1)
        
        return torch.cat(generation_tokens, dim=1)  # [bs, num_generated_tokens, output_dim]
