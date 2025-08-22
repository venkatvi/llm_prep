"""
Copyright (c) 2025. All rights reserved.
"""

"""
Multi-head self-attention mechanism for transformer models.
"""

import torch 
import math 

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor)->torch.Tensor: 
    sqrt_d =  math.sqrt(q.size(-1))
    scores = q @ k.transpose(-2, -1) # bs, n_heads, seq_len, head_dim @ bs, n_heads, head_dim, seq_len
    scores = scores/sqrt_d # bs, n_heads, seq_len, seq_len
    scores = torch.softmax(scores, dim=-1) # compute softmax scores along the last dim 

    attn_out = scores @ v # bs, n_heads, seq_len, seq_len @ bs, n_heads, seq_len, head_dim --> bs, n_heads, seq_len, head_dim
    attn_out = attn_out.transpose(1, 2) # bs, seq_len, n_heads, head_dim 
    return attn_out

class MultiHeadAttention(torch.nn.Module):
    """Multi-head self-attention with scaled dot-product attention."""
    def __init__(self, embed_dim: int, num_heads: int): 
        super().__init__()
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        assert embed_dim % num_heads == 0, "embedding dim should be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads 
        self.sqrt_d = math.sqrt(self.head_dim)

        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, input: torch.Tensor)->torch.Tensor: 
        """Apply multi-head self-attention to input tensor."""
        # input_size = (batch_size, seq_len, embed_dim)
        q = self.q_proj(input).reshape([input.size(0), input.size(1), self.num_heads, self.head_dim]).permute(0,2,1,3)
        k = self.k_proj(input).reshape([input.size(0), input.size(1), self.num_heads, self.head_dim]).permute(0,2,1,3)
        v = self.v_proj(input).reshape([input.size(0), input.size(1), self.num_heads, self.head_dim]).permute(0,2,1,3)

        # scaled dot-product attention 
        attn_out = scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.reshape([input.size(0), input.size(1), self.embed_dim])
        return self.out_proj(attn_out)



        
