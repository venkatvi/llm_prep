import torch

from transformer.attention import MultiHeadAttention
from transformer.ffn import FFN


class Decoder(torch.nn.Module):
    """Single transformer decoder layer with self-attention, cross-attention, and FFN.

    Implements the standard transformer decoder layer with:
    1. Masked self-attention (causal masking)
    2. Cross-attention with encoder outputs
    3. Feed-forward network
    Each sub-layer has residual connections and layer normalization.

    Attributes:
        self_attention (MultiHeadAttention): Masked self-attention layer
        cross_attention (MultiHeadAttention): Cross-attention with encoder
        ffn (FFN): Feed-forward network
        layer_norm_1 (torch.nn.LayerNorm): Layer norm after self-attention
        layer_norm_2 (torch.nn.LayerNorm): Layer norm after cross-attention
        layer_norm_3 (torch.nn.LayerNorm): Layer norm after FFN
    """

    def __init__(self, embed_dim: int, num_heads: int, latent_dim: int) -> None:
        """Initialize transformer decoder layer.

        Args:
            embed_dim (int): Embedding dimension for the model
            num_heads (int): Number of attention heads
            latent_dim (int): Hidden dimension in feed-forward network
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, apply_causal_mask=True
        )
        self.cross_attention = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, apply_causal_mask=False
        )
        self.ffn = FFN(embed_dim=embed_dim, latent_dim=latent_dim)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_3 = torch.nn.LayerNorm(embed_dim)

    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder layer.

        Processes decoder input through three sub-layers:
        1. Masked self-attention (prevents looking at future tokens)
        2. Cross-attention with encoder outputs (for conditioning on source)
        3. Feed-forward network

        Args:
            decoder_input (torch.Tensor): Decoder input [batch_size, tgt_len, embed_dim]
            encoder_output (torch.Tensor): Encoder outputs [batch_size, src_len, embed_dim]

        Returns:
            torch.Tensor: Decoder output [batch_size, tgt_len, embed_dim]
        """
        # 1. Masked self-attention
        x = self.layer_norm_1(decoder_input + self.self_attention(decoder_input))

        # 2. Full cross attention
        x = self.layer_norm_2(x + self.cross_attention(x, encoder_output))

        # 3. FFN
        x = self.layer_norm_3(x + self.ffn(x))
        return x
