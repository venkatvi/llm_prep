"""
Copyright (c) 2025. All rights reserved.
"""

"""
Transformer encoder model for sequence processing.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.decoder import Decoder
from transformer.encoder import Encoder
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
        max_seq_len: int,
        attention_type: str,
    ) -> None:
        """Initialize transformer model for regression tasks.

        Args:
            input_dim (int): Dimension of input features
            embed_dim (int): Model embedding dimension
            ffn_latent_dim (int): Hidden dimension in feed-forward layers
            num_layers (int): Number of transformer encoder layers
            num_heads (int): Number of attention heads in each layer
            output_dim (int): Final output dimension
            apply_causal_mask (bool): Whether to apply causal masking in attention
            max_seq_len (int): Maximum sequence length for positional encoding
        """
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, embed_dim)
        self.pe = PositionalEncoding(seq_len=max_seq_len, d_model=embed_dim)
        self.layers = torch.nn.ModuleList(
            [
                Encoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_latent_dim=ffn_latent_dim,
                    apply_causal_mask=apply_causal_mask,
                )
                for _ in range(num_layers)
            ]
        )
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
        max_seq_len: int,
        attention_type: str
    ) -> None:
        """Initialize autoregressive transformer model.

        Args:
            input_dim (int): Dimension of input features
            embed_dim (int): Model embedding dimension
            ffn_latent_dim (int): Hidden dimension in feed-forward layers
            num_layers (int): Number of transformer encoder layers
            num_heads (int): Number of attention heads in each layer
            output_dim (int): Final output dimension
            apply_causal_mask (bool): Whether to apply causal masking in attention
            max_seq_len (int): Maximum sequence length for positional encoding
        """
        super().__init__(
            input_dim=input_dim,
            embed_dim=embed_dim,
            ffn_latent_dim=ffn_latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=output_dim,
            apply_causal_mask=apply_causal_mask,
            max_seq_len=max_seq_len,
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


class EncoderDecoder(torch.nn.Module):
    """Encoder-Decoder transformer architecture for sequence-to-sequence tasks.

    Implements the classic transformer architecture with separate encoder and decoder
    stacks for tasks like machine translation, text summarization, etc.

    Attributes:
        encoder_input_proj (torch.nn.Linear): Input projection for encoder
        decoder_input_proj (torch.nn.Linear): Input projection for decoder
        encoder_pe (PositionalEncoding): Positional encoding for encoder
        decoder_pe (PositionalEncoding): Positional encoding for decoder
        encoder_layers (torch.nn.ModuleList): List of encoder layers
        decoder_layers (torch.nn.ModuleList): List of decoder layers
        out_proj (torch.nn.Linear): Final output projection
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        ffn_latent_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        output_dim: int,
        apply_causal_mask: bool,
        max_seq_len: int,
        attention_type: str 
    ) -> None:
        """Initialize encoder-decoder transformer.

        Args:
            input_dim (int): Dimension of input features
            embed_dim (int): Model embedding dimension
            ffn_latent_dim (int): Hidden dimension in feed-forward layers
            num_encoder_layers (int): Number of encoder layers
            num_decoder_layers (int): Number of decoder layers
            num_heads (int): Number of attention heads in each layer
            output_dim (int): Final output dimension
            apply_causal_mask (bool): Whether to apply causal masking in decoder
            max_seq_len (int): Maximum sequence length for positional encoding
        """
        super().__init__()

        self.encoder_input_proj = torch.nn.Linear(input_dim, embed_dim)
        self.decoder_input_proj = torch.nn.Linear(input_dim, embed_dim)

        self.encoder_pe = PositionalEncoding(seq_len=max_seq_len, d_model=embed_dim)
        self.decoder_pe = PositionalEncoding(seq_len=max_seq_len, d_model=embed_dim)

        self.encoder_layers = torch.nn.ModuleList(
            [
                Encoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_latent_dim=ffn_latent_dim,
                    apply_causal_mask=apply_causal_mask,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_layers = torch.nn.ModuleList(
            [
                Decoder(embed_dim=embed_dim, num_heads=num_heads, latent_dim=ffn_latent_dim)
                for _ in range(num_decoder_layers)
            ]
        )

        self.out_proj = torch.nn.Linear(embed_dim, output_dim)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """Encode input sequence using transformer encoder stack.

        Args:
            input (torch.Tensor): Input sequence [batch_size, src_len, input_dim]

        Returns:
            torch.Tensor: Encoded representation [batch_size, src_len, embed_dim]
        """
        x = self.encoder_input_proj(input)
        x = self.encoder_pe(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, input: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """Decode target sequence using transformer decoder stack with encoder outputs.

        Args:
            input (torch.Tensor): Target sequence input [batch_size, tgt_len, input_dim]
            encoder_output (torch.Tensor): Encoder outputs [batch_size, src_len, embed_dim]

        Returns:
            torch.Tensor: Decoded sequence [batch_size, tgt_len, output_dim]
        """
        x = self.decoder_input_proj(input)
        x = self.decoder_pe(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output)

        return self.out_proj(x)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder-decoder transformer.

        Args:
            encoder_input (torch.Tensor): Source sequence [batch_size, src_len, input_dim]
            decoder_input (torch.Tensor): Target sequence [batch_size, tgt_len, input_dim]

        Returns:
            torch.Tensor: Decoded output sequence [batch_size, tgt_len, output_dim]
        """
        encoder_output = self.encode(encoder_input)
        decoder_output = self.decode(decoder_input, encoder_output)
        return decoder_output
