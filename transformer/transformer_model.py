import torch 
from attention import MultiHeadAttention
from ffn import FFN
from input_encodings import PositionalEncoding
class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_latent_dim: int): 
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FFN(embed_dim=embed_dim, latent_dim=ffn_latent_dim)
        self.norm_1 = torch.nn.LayerNorm(embed_dim)
        self.norm_2 = torch.nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.norm_1(x + self.attn(x)) # post-norm
        x = self.norm_2(x + self.ffn(x)) # psot-norm
        return x 
    
MAX_SEQUENCE_LENGTH = 128000
class TransformerModel(torch.nn.Module): 
    def __init__(self, input_dim: int, embed_dim: int, ffn_latent_dim:int, num_layers:int, num_heads: int, output_dim: int):
        self.input_proj = torch.nn.Linear(input_dim, embed_dim)
        self.pe = PositionalEncoding(seq_len = MAX_SEQUENCE_LENGTH, d_model = embed_dim)
        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, ffn_latent_dim=ffn_latent_dim)
            ] for _ in range(num_layers)
        )
        self.out_proj = torch.nn.Linear(embed_dim, output_dim)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        x = self.input_proj(input)
        x = self.pe(x)
        for layer_idx in len(self.layers):
            x = self.layers[layer_idx](x)
        # x 's dim. = bs, seq_len, embed_dim --> bs, embed_dim
        x = x.mean(dim=1) # global average pooling over sequence length 
        return self.out_proj(x) # bs, outputdim 


