"""
Copyright (c) 2025. All rights reserved.
"""

"""
Feedforward network for transformer models.
"""

import torch


class FFN(torch.nn.Module):
    """Two-layer feedforward network with ReLU activation.

    Standard transformer FFN that expands to a larger dimension in the middle layer
    and projects back to the original dimension. Uses ReLU activation between layers.

    Attributes:
        layer_1 (torch.nn.Linear): First linear layer (embed_dim -> latent_dim)
        relu (torch.nn.ReLU): ReLU activation function
        layer_2 (torch.nn.Linear): Second linear layer (latent_dim -> embed_dim)
    """

    layer_1: torch.nn.Linear
    relu: torch.nn.ReLU
    layer_2: torch.nn.Linear

    def __init__(self, embed_dim: int, latent_dim: int) -> None:
        """Initialize feedforward network.

        Args:
            embed_dim (int): Input/output embedding dimension
            latent_dim (int): Hidden layer dimension (typically 4x embed_dim)
        """
        super().__init__()
        self.layer_1 = torch.nn.Linear(embed_dim, latent_dim)
        self.relu = torch.nn.ReLU()
        self.layer_2 = torch.nn.Linear(latent_dim, embed_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply two-layer feedforward transformation.

        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        return self.layer_2(self.relu(self.layer_1(input)))
