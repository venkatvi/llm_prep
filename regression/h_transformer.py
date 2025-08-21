import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
from transformer.transformer_model import TransformerModel
from dataclasses import dataclass 
from configs import TransformerModelConfig

from typing import Optional, Tuple 
class TransformerRegressionModel(torch.nn.Module): 
    def __init__(self, config: TransformerModelConfig): 
        super().__init__()
        self.config = config
        self.model = TransformerModel(
            input_dim=config.input, 
            embed_dim=config.embed_dim, 
            ffn_latent_dim=config.ffn_latent_dim, 
            num_layers=config.num_layers, 
            num_heads=config.num_heads, 
            output_dim=config.output_dim
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.model(x)

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]: 
        if random_seed: 
            import random
            import numpy as np 
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # Generate 100 samples of input_dim=8 
        num_samples = 100 
        sequence_length = 32
        input_dim = self.config.input_dim
        x = torch.rand([batch_size, sequence_length, input_dim]) 
        y = torch.sum(x.reshape([batch_size, sequence_length * input_dim]), dim=1)
        return x, y    

if __name__ == "__main__": 
    model = TransformerModel(
        input_dim=8,
        embed_dim=32, 
        ffn_latent_dim=128, 
        num_layers=2, 
        num_heads=2, 
        output_dim=1
    )
    model.eval()
    output = model(input)