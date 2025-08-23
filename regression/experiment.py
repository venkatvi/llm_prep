import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
from typing import Optional
from dataset import prepare_data
from e_linear_reg import LinearRegressionModel
from e_non_linear_reg import MLP
from h_transformer import TransformerRegressionModel
from lib.experiment import Experiment
from lib.configs import ExperimentConfig
from lib.train import train_model, predict_model, train_model_with_dataloader, split_data
from lib.utils import plot_results

class RegressionExperiment(Experiment): 
    def __init__(self, config: ExperimentConfig) -> None: 
        super().__init__(config)
        self.model = self.define_model()
        # Generate synthetic training data
        self.inputs, self.targets = self.model.generate_data(self.config.data.fix_random_seed)
        
    
    def define_model(self) -> torch.nn.Module:
        """Create regression model based on experiment type."""
        if self.config.type == "linear":
            return LinearRegressionModel(self.config.model)
        elif self.config.type == "transformer":
            return TransformerRegressionModel(self.config.model)
        else:
            return MLP(self.config.model)
    
    def train(self) -> None:
        """Execute training with DataLoader or direct tensor processing.""" 
        # Choose training mode: DataLoader vs direct tensor processing
        if self.config.data.use_dataloader: 
            # Split data for DataLoader training
            train_inputs, train_targets, val_inputs, val_targets = split_data(self.inputs, self.targets)
            
            # Create DataLoaders and temporary CSV files
            train_dataloader, train_dataset_file_name = prepare_data(train_inputs, train_targets, suffix="_train.csv", batch_size=self.config.data.training_batch_size)
            val_dataloader, val_dataset_file_name= prepare_data(val_inputs, val_targets, suffix="_val.csv")
            
            try: 
                # Train with DataLoader (batch processing)
                self.train_loss, self.val_loss = train_model_with_dataloader(
                    self.model, 
                    self.train_context, 
                    train_dataloader, 
                    val_dataloader
                )
            except:
                
                # Clean up temporary files on error
                os.remove(train_dataset_file_name)
                os.remove(val_dataset_file_name)
                raise RuntimeError("Training Failed ")
            
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
    def __init__(self, config: ExperimentConfig): 
        super().__init__(config)
        self.decode_config = config.model.decode_config
    
    def predict_autoregressively(self, input: torch.Tensor, num_steps_override: Optional[int]=None) -> torch.Tensor: 
        # input = [bs, seqlen, embed_dim]
        generation_tokens = []
        if num_steps_override is None: 
            num_steps_override = self.decode_config.num_steps

        current_input = input.clone()
        with torch.no_grad():
            for step in range(num_steps_override): 
                print(f"Generating next_token in {step}, current_input size: {current_input.size()}")
                output = self.model(current_input) # transformer_model(input) --> bs, output_dim 
                next_token = output.unsqueeze(dim=-1) # since the model collects mean along Sequence length

                generation_tokens.append(next_token)
                
                if self.decode_config.expanding_context: 
                    current_input = torch.cat([
                        current_input, 
                        next_token, 
                    ], dim=1)

                    if current_input.size(1) > self.decode_config.max_seq_len: 
                        current_input = current_input[:, -self.decode_config.max_seq_len:, :]
                else:
                    current_input = torch.cat([
                        current_input[:, 1:, :], # bs, seq_lem, input_dim=1
                        next_token, # bs, output_dim, 1
                    ], dim=1)
        
        return torch.cat(generation_tokens, dim=1) # bs, num_generated_tokens, 1
