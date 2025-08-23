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
from transformer.encoder import Encoder
from transformer.ffn import FFN
from transformer.input_encodings import PositionalEncoding

class TransformerModel(torch.nn.Module):
    """Complete transformer encoder model with positional encoding.""" 
    def __init__(self, input_dim: int, embed_dim: int, ffn_latent_dim:int, num_layers:int, num_heads: int, output_dim: int, apply_causal_mask: bool, max_seq_len: int):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, embed_dim)
        self.pe = PositionalEncoding(seq_len = max_seq_len, d_model = embed_dim)
        self.layers = torch.nn.ModuleList([
                Encoder(embed_dim=embed_dim, num_heads=num_heads, ffn_latent_dim=ffn_latent_dim, apply_causal_mask=apply_causal_mask) for _ in range(num_layers)
        ]) 
        self.out_proj = torch.nn.Linear(embed_dim, output_dim)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        """Process input through transformer layers with global average pooling."""
        x = self.input_proj(input)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        # x's dim: bs, seq_len, embed_dim -> bs, embed_dim
        x = x.mean(dim=1) # global average pooling over sequence length 
        return self.out_proj(x) # bs, embed_dim --> bs, output_dim 


