"""
Copyright (c) 2025. All rights reserved.
"""

"""
Feedforward network for transformer models.
"""

import torch 

class FFN(torch.nn.Module):
    """Two-layer feedforward network with ReLU activation."""
    def __init__(self, embed_dim: int, latent_dim: int):
        super().__init__()
        self.layer_1 = torch.nn.Linear(embed_dim, latent_dim)
        self.relu = torch.nn.ReLU()
        self.layer_2 = torch.nn.Linear(latent_dim, embed_dim)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        """Apply two-layer feedforward transformation."""
        return self.layer_2(self.relu(self.layer_1(input)))
    