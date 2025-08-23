"""
Copyright (c) 2025. All rights reserved.
"""

"""
Transformer encoder model for sequence processing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import List
from transformer.encoder import Encoder
from transformer.ffn import FFN
from transformer.input_encodings import PositionalEncoding

class TransformerModel(torch.nn.Module):
    """Complete transformer encoder model with positional encoding for regression tasks.
    
    This model applies global average pooling over the sequence dimension to produce
    a single output per sample, making it suitable for regression where we need to
    predict a scalar value from a sequence input.
    """
    
    input_proj: torch.nn.Linear
    pe: PositionalEncoding
    layers: torch.nn.ModuleList
    out_proj: torch.nn.Linear
    
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        ffn_latent_dim: int, 
        num_layers: int, 
        num_heads: int, 
        output_dim: int, 
        apply_causal_mask: bool, 
        max_seq_len: int
    ) -> None:
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, embed_dim)
        self.pe = PositionalEncoding(seq_len=max_seq_len, d_model=embed_dim)
        self.layers = torch.nn.ModuleList([
            Encoder(embed_dim=embed_dim, num_heads=num_heads, ffn_latent_dim=ffn_latent_dim, apply_causal_mask=apply_causal_mask)
            for _ in range(num_layers)
        ]) 
        self.out_proj = torch.nn.Linear(embed_dim, output_dim)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Process input through transformer layers with global average pooling.
        
        Args:
            input: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor [batch_size, output_dim] - pooled representation
        """
        # Project input to embedding dimension
        x: torch.Tensor = self.input_proj(input)  # [bs, seq_len, embed_dim]
        # Add positional encoding
        x = self.pe(x)  # [bs, seq_len, embed_dim]
        # Pass through transformer encoder layers
        for layer in self.layers:
            x = layer(x)  # [bs, seq_len, embed_dim]
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [bs, embed_dim]
        # Final projection to output dimension
        return self.out_proj(x)  # [bs, output_dim]


class AutoregressiveTransformerModel(TransformerModel):
    """Autoregressive transformer model for sequence-to-sequence generation.
    
    Unlike the base TransformerModel, this variant does not apply global average pooling
    and instead returns the full sequence representation, enabling token-by-token generation
    for autoregressive tasks.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        ffn_latent_dim: int,
        num_layers: int,
        num_heads: int,
        output_dim: int,
        apply_causal_mask: bool,
        max_seq_len: int
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            embed_dim=embed_dim,
            ffn_latent_dim=ffn_latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=output_dim,
            apply_causal_mask=apply_causal_mask,
            max_seq_len=max_seq_len
        )
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Process input through transformer layers and return full sequence.
        
        Args:
            input: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim] - full sequence representation
        """
        # Project input to embedding dimension
        x: torch.Tensor = self.input_proj(input)  # [bs, seq_len, embed_dim]
        # Add positional encoding
        x = self.pe(x)  # [bs, seq_len, embed_dim]
        # Pass through transformer encoder layers with causal masking
        for layer in self.layers:
            x = layer(x)  # [bs, seq_len, embed_dim]
        # Project to output dimension (no pooling for autoregressive)
        return self.out_proj(x)  # [bs, seq_len, output_dim]
    def generate_next_token(self, input: torch.Tensor) -> torch.Tensor:
        """Generate the next token in the sequence.
        
        Args:
            input: Current sequence [batch_size, seq_len, input_dim]
            
        Returns:
            Next token prediction [batch_size, 1, output_dim]
        """
        # Get full sequence output
        output: torch.Tensor = self.forward(input)  # [bs, seq_len, output_dim]
        # Return only the last token for next-token prediction
        return output[:, -1:, :]  # [bs, 1, output_dim]