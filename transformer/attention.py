import torch 
import math

class MultiHeadSelfAttention(torch.nn.Module): 
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension should be equally divisible by n heads"
        self.embed_dim = embed_dim # d_model
        self.n_heads = n_heads 
        self.head_dim = embed_dim//n_heads 

        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = torch.nn.Linear(self.head_dim, self.embed_dim)

        # Alternatively - 1 linear op a big one 
        # self.qkvproj = torch.nn.Linear(self.embed_dim, 3*self.embed_dim)
        
    def forward(self, input: torch.Tensor)->torch.Tensor: 
        # input.shape = (batchsize, seq_len, d_model aka embed_dim)
        x = input.reshape(input.size(0)*input.size(1), input.size(2))
        desired_dim = [input.size(0), input.size(1), self.n_heads, self.head_dim]
        q = self.q_proj(x).reshape(desired_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(desired_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(desired_dim).permute(0, 2, 1, 3)
        
        # Alternatively: 
        # qkv = self.qkvproj(x) # (bs*seq_len, 3*embed_dim)
        # qkv = qkv.reshape(bs, seq_len, 3, embed_dim).permute(2, 0, 1, 3)
        # q = qkv[0]
        # k = qkv[1]
        # v = qkv[2] (bs, seq_len, embed_dim) --> (bs, seq_len, num_heads, head_dim) --> (bs, num_heads, seq_len, head_dim)

        ## Attention
        qk = q @ k.Transpose(-2, -1) # bs, num_heads, seq_len, heads_dim @ bs, num_heads, heads_dim, seq_len 
        scores = qk/math.sqrt(self.head_dim) # bs, num_heads, seq_len, seq_len

        # The softmax is applied over the last dim (seq_len), so attention weights sum to 1 across the last axis for each query position.
        attn_weights = torch.softmax(scores, dim=-1) # bs, num_heads, seq_len, seq_len

        attn_out = attn_weights @ v # bs, num_heads, seq_len, seq_len * bs, num_heads, seq_len, heads_dim --> bs, num_heads, seq_len, heads_dim
        attn_out = attn_out.Transpose(1, 2).reshape([input.size(0), input.size(1), self.embed_dim])
        return self.out_proj(attn_out)


        
           
        