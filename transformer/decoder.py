import torch

from transformer.ffn import FFN
from transformer.attention_utils import ATTENTION_TYPE, get_attention

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

    def __init__(self, embed_dim: int, num_heads: int, num_groups: int, latent_dim: int, attention_type: str, use_kv_cache: bool = False) -> None:
        """Initialize transformer decoder layer.

        Args:
            embed_dim (int): Embedding dimension for the model
            num_heads (int): Number of attention heads
            latent_dim (int): Hidden dimension in feed-forward network
            attention_type (str): Type of attention mechanism
            use_kv_cache (bool): Whether to use KV caching
        """
        super().__init__()

        self.self_attention = get_attention(
            attention_type=attention_type,
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_groups=num_groups,
            apply_causal_mask=True,
            use_kv_cache=use_kv_cache
        )
        self.cross_attention = get_attention(
            attention_type=attention_type,
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_groups=num_groups,
            apply_causal_mask=False,
            use_kv_cache=use_kv_cache
        )
        self.ffn = FFN(embed_dim=embed_dim, latent_dim=latent_dim)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_3 = torch.nn.LayerNorm(embed_dim)

    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor, expanding_context: bool = True) -> torch.Tensor:
        """Forward pass through decoder layer.

        Processes decoder input through three sub-layers:
        1. Masked self-attention (prevents looking at future tokens)
        2. Cross-attention with encoder outputs (for conditioning on source)
        3. Feed-forward network

        Args:
            decoder_input (torch.Tensor): Decoder input [batch_size, tgt_len, embed_dim]
            encoder_output (torch.Tensor): Encoder outputs [batch_size, src_len, embed_dim]
            expanding_context (bool): Cache expansion mode for attention

        Returns:
            torch.Tensor: Decoder output [batch_size, tgt_len, embed_dim]
        """
        # 1. Masked self-attention
        x = self.layer_norm_1(decoder_input + self.self_attention(decoder_input, kv=None, expanding_context=expanding_context))

        # 2. Cross-attention with encoder output as key/value
        x = self.layer_norm_2(x + self.cross_attention(x, kv=encoder_output, expanding_context=expanding_context))

        # 3. FFN
        x = self.layer_norm_3(x + self.ffn(x))
        return x
