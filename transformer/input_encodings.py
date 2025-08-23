"""
Copyright (c) 2025. All rights reserved.
"""

"""
Positional encoding for transformer models.
"""

import torch


class PositionalEncoding(torch.nn.Module):
    """Sinusoidal positional encoding for sequence position information.

    Implements the sinusoidal positional encoding from the "Attention is All You Need" paper.
    Uses sine and cosine functions of different frequencies to encode position information.
    The encoding is added to input embeddings to provide sequence position context.

    Attributes:
        seq_len (int): Maximum sequence length
        d_model (int): Model embedding dimension
        pe (torch.Tensor): Precomputed positional encoding matrix
    """

    seq_len: int
    d_model: int

    def __init__(self, seq_len: int, d_model: int) -> None:
        """Initialize positional encoding.

        Precomputes sinusoidal positional encodings for all positions up to seq_len.
        The encoding uses the formula:
        - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            seq_len (int): Maximum sequence length to support
            d_model (int): Model embedding dimension (must be even)
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # pe(pos, 2i) = sin(pos/ 10000^2i/d)
        # pe(pos, 2i+1) = cos(pos/ 10000^2i/d)
        pe: torch.Tensor = torch.zeros([self.seq_len, self.d_model])

        pos: torch.Tensor = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(
            dim=1
        )  # (seq_len, 1)
        idx: torch.Tensor = torch.arange(self.d_model, dtype=torch.float).unsqueeze(
            dim=0
        )  # (1, d_model)

        angle_rates: torch.Tensor = 1 / 10000 ** (2 * (idx // 2) / d_model)
        angle_rads: torch.Tensor = pos * angle_rates

        pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])  # even indices
        pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])  # odd indices

        pe = pe.unsqueeze(0)  # Shape: [1, seq_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            input (torch.Tensor): Input embeddings [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Input embeddings with added positional encoding
                        [batch_size, seq_len, d_model]
        """
        seq_len: int = input.size(1)  # (batch_size, seq_len, d_model)
        return input + self.pe[:, :seq_len, :]
