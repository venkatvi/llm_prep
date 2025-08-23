import torch

from transformer.attention import MultiHeadAttention
from transformer.ffn import FFN

class Decoder(torch.nn.Module): 
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int,
            latent_dim: int 
        ) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            apply_causal_mask=True
        )
        self.cross_attention = MultiHeadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            apply_causal_mask=False
        )
        self.ffn = FFN(embed_dim=embed_dim, latent_dim=latent_dim)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_3 = torch.nn.LayerNorm(embed_dim)
    
    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor: 
        # 1. Masked self-attention 
        x = self.layer_norm_1(decoder_input + self.self_attention(decoder_input))

        # 2. Full cross attention
        x = self.layer_norm_2(x + self.cross_attention(x, encoder_output))

        # 3. FFN 
        x = self.layer_norm_3(x + self.ffn(x))
        return x 

