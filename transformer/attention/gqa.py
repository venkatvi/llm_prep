import torch 
import math 
class GroupQueryAttention(torch.nn.Module): 
    def __init__(
            self,
            embed_dim: int, 
            num_heads: int, 
            num_groups: int 
    ): 
        super().__init__()
        assert(embed_dim % num_heads == 0 , "Embed dim should be divisible by num_heads")
        assert(embed_dim % num_groups == 0, "Embed dim should be divisible by num_groups")
        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_groups

        assert (self.num_groups < self.num_heads)
        
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, self.num_groups * self.head_dim)
        self.v_proj = torch.nn.Linear(embed_dim, self.num_groups * self.head_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        # input = bs, slen, embeddim

        q = self.q_proj(input).reshape([input.size(0), input.size(1), self.num_heads, self.head_dim]).permute(0,2,1,3) # bs, num_heads, slen, head_dim
        k = self.k_proj(input).reshape([input.size(0), input.size(1), self.num_groups, self.head_dim]).permute(0,2,1,3) # bs, num_groups, slen, head_dim 
        v = self.v_proj(input).reshape([input.size(0), input.size(1), self.num_groups, self.head_dim]).permute(0,2,1,3) 
        
        k = k.repeat_interleave(self.group_size, dim=1) # along each group--> create group_size copies to cover for all the num_heads
        v = v.repeat_interleave(self.group_size, dim=1) 

        scores = q @ k.transpose(-2, -1)
        scores = scores /math.sqrt(self.head_dim)
        scores = torch.softmax(scores, dim=-1)

        attn_out = scores @ v # bs, nheads, slen, slen @ bs, nheads, slen, head_dim --> bs, nheads, slen, headim
        attn_out = attn_out.transpose(1, 2).reshape([input.size(0), input.size(1), input.size(2)])

        return self.out_proj(attn_out)



