import torch 
import math 
class MultiQueryAttention(torch.nn.Module): 
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int 
        ):
        super().__init__()
        assert ( embed_dim % num_heads == 0, "Embed dim should be divisible by num_heads." )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, self.head_dim)
        self.v_proj = torch.nn.Linear(embed_dim, self.head_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        # input [bs, slen, embed_dim]
        
        q = self.q_proj(input).reshape([input.size(0), input.size(1), self.num_heads, self.head_dim]).permute(0, 2, 1, 3) # bs, nheads, slen, headdim
        k = self.k_proj(input).unsqueeze(1) # bs, 1, slen, headdim
        v = self.v_proj(input).unsqueeze(1) # bs, 1, slen, headdim # broadcasting is implicit if no expand is needed.
        # v = v.expand(input.size(0), self.num_heads,  input.size(1), self.head_dim)

        # walk through SDPA 
        sqrt_d: float = math.sqrt(self.head_dim)
        scores = q @ k.transpose(-2, -1) # bs, nheads, slen, headdim @ bs, 1, headdim, slen 
        scores = scores / sqrt_d # bs, nheads, slen, 

        attn_scores = torch.softmax(scores, dim=-1) # 

        attn_out = attn_scores @ v # bs, nheads, slen, slen @ bs, 1, slen, headdim --> bs, nheads, slen, headdim
        attn_out = attn_out.permute(0, 2, 1, 3 ).reshape([input.size(0), input.size(1), input.size(2)])
        
        out = self.out_proj(attn_out)
        return out 


    