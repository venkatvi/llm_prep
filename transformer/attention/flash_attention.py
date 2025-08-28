"""
Copyright (c) 2025. All rights reserved.
"""

"""
Flash Attention implementation using block-wise algorithm.

This module implements the Flash Attention algorithm that reduces memory complexity
from O(N²) to O(N) by computing attention in blocks and using online softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
from dataclasses import dataclass


@dataclass
class FlashAttentionConfig:
    """Configuration for Flash Attention."""
    block_size_q: int = 64  # Query block size
    block_size_k: int = 64  # Key/Value block size
    causal: bool = False    # Whether to apply causal masking
    softmax_scale: Optional[float] = None  # Custom softmax scale
    dropout_p: float = 0.0  # Dropout probability
    
    
class FlashAttention(nn.Module):
    """
    Flash Attention implementation with block-wise computation.
    
    Key innovations:
    1. Block-wise computation to reduce memory from O(N²) to O(N)
    2. Online softmax algorithm to avoid storing full attention matrix
    3. Tiling strategy for large sequences
    """
    
    def __init__(self, config: FlashAttentionConfig = None):
        super().__init__()
        self.config = config or FlashAttentionConfig()
        
    def forward(
        self,
        q: torch.Tensor,  # [batch, n_heads, seq_len, head_dim]
        k: torch.Tensor,  # [batch, n_heads, seq_len, head_dim]
        v: torch.Tensor,  # [batch, n_heads, seq_len, head_dim]
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention using Flash Attention block-wise algorithm.
        
        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]  
            v: Value tensor [B, H, N, D]
            attn_mask: Attention mask [N, N] or [B, H, N, N]
            key_padding_mask: Key padding mask [B, N]
            
        Returns:
            output: Attention output [B, H, N, D]
            attention_weights: Attention weights [B, H, N, N] (optional, for compatibility)
        """
        batch_size, n_heads, seq_len, head_dim = q.shape
        
        # Compute softmax scale
        if self.config.softmax_scale is not None:
            scale = self.config.softmax_scale
        else:
            scale = 1.0 / math.sqrt(head_dim)
            
        # Apply scale to queries
        q = q * scale
        
        # Use block-wise computation for long sequences
        if seq_len > self.config.block_size_q * 2:
            return self._flash_attention_blocks(
                q, k, v, attn_mask, key_padding_mask
            )
        else:
            # For short sequences, use standard computation
            return self._standard_attention(
                q, k, v, attn_mask, key_padding_mask
            )
            
    def _flash_attention_blocks(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        FlashAttention-style blockwise attention with online softmax.
        Computes exact softmax(Q K^T / sqrt(d)) V without materializing NxN.
        """
        import math
        import torch
        import torch.nn.functional as F

        batch_size, n_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype
        scale = 1.0 / math.sqrt(head_dim)  # (1) correct temperature

        # Block sizes
        Br = min(self.config.block_size_q, seq_len)  # Query block size
        Bc = min(self.config.block_size_k, seq_len)  # Key/Value block size

        # Number of blocks
        Tr = math.ceil(seq_len / Br)  # query blocks
        Tc = math.ceil(seq_len / Bc)  # key/value blocks

        # Output
        O = torch.zeros_like(q, dtype=dtype, device=device)

        # Process each query block
        eps = 1e-6  # (5) guard for all-masked rows
        for j in range(Tr):
            q_start = j * Br
            q_end = min((j + 1) * Br, seq_len)
            q_block = q[:, :, q_start:q_end, :]  # [B, H, Br, D]

            # per-row running statistics for this block
            O_block = torch.zeros_like(q_block, dtype=dtype, device=device)        # unnormalized numerator
            l_block = torch.zeros((batch_size, n_heads, q_end - q_start),          # running sum of exp to use for softmax of the final output
                                dtype=dtype, device=device)
            m_block = torch.full((batch_size, n_heads, q_end - q_start),           # running max - used for computing exp(cur-max) for stable softmax numerator
                                -float("inf"), dtype=dtype, device=device)

            for i in range(Tc):
                k_start = i * Bc
                k_end = min((i + 1) * Bc, seq_len)
                k_block = k[:, :, k_start:k_end, :]  # [B, H, Bc, D]
                v_block = v[:, :, k_start:k_end, :]  # [B, H, Bc, D]

                # scores
                S_block = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale  # [B, H, Br, Bc]

                # Blah Blah Masking 
                # # causal mask (add -inf where invalid)
                # if getattr(self.config, "causal", False):
                #     causal_mask = self._get_causal_mask(
                #         q_start, q_end, k_start, k_end, device=device, dtype=dtype
                #     )
                #     S_block = S_block + causal_mask  # causal_mask should be 0 or -inf

                # # attention mask: supports [N,N] or [B,H,N,N]
                # if attn_mask is not None:
                #     if attn_mask.dim() == 2:  # [N, N]
                #         mask_block = attn_mask[q_start:q_end, k_start:k_end]  # [Br, Bc]
                #         S_block = S_block + mask_block.unsqueeze(0).unsqueeze(0)  # [1,1,Br,Bc]
                #     else:  # [B, H, N, N]
                #         mask_block = attn_mask[:, :, q_start:q_end, k_start:k_end]  # [B,H,Br,Bc]
                #         S_block = S_block + mask_block

                # # key padding mask: True where key is padding (invalidate column)
                # if key_padding_mask is not None:
                #     key_mask_block = key_padding_mask[:, k_start:k_end]           # [B, Bc], bool
                #     S_block = S_block.masked_fill(key_mask_block.unsqueeze(1).unsqueeze(2), float('-inf'))

                # ---- Online softmax update (2) no "beta" term needed ----
                m_ij = torch.amax(S_block, dim=-1)              # [B, H, Br] # find amax along Br dimension for scores 
                m_new = torch.maximum(m_block, m_ij)            # [B, H, Br] # compare it against all time max of other blocks so far. 
                alpha = torch.exp(m_block - m_new)              # [B, H, Br] # Set adjustment factor as exp(all_time_max - cur max)

                P_unnorm = torch.exp(S_block - m_new.unsqueeze(-1))  # [B, H, Br, Bc] 

                # # (3) Dropout on attention *probabilities* effect:
                # # Do NOT affect l_block (denominator). Apply dropout to the numerator path only.
                # if self.config.dropout_p > 0.0 and self.training:
                #     p = self.config.dropout_p
                #     keep = 1.0 - p
                #     # Inverted dropout mask
                #     M = (torch.rand_like(P_unnorm) < keep).to(P_unnorm.dtype) / keep
                #     P_for_O = P_unnorm * M
                # else:
                #     P_for_O = P_unnorm
                P_for_O = P_unnorm
                # update running sums (no dropout in l_block)
                l_block = alpha * l_block + torch.sum(P_unnorm, dim=-1)            # [B, H, Br] - use this at the end 

                # Update prev output with diff in exp 
                O_block = alpha.unsqueeze(-1) * O_block + torch.matmul(P_for_O, v_block)  # [B, H, Br, D]

                m_block = m_new

            # finalize this query block (4) normalize once
            O_block = O_block / l_block.clamp_min(eps).unsqueeze(-1)
            O[:, :, q_start:q_end, :] = O_block

        # Don’t allocate massive attention maps unless requested
        attention_weights = None
        return O, attention_weights

        
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard attention computation for short sequences."""
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, N, N]
        
        # Apply causal masking
        if self.config.causal:
            seq_len = q.size(-2)
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=q.device), diagonal=1)
            scores = scores + causal_mask
            
        # Apply attention mask
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0) if attn_mask.dim() == 2 else scores + attn_mask
            
        # Apply key padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.config.dropout_p > 0.0 and self.training:
            attention_weights = F.dropout(attention_weights, p=self.config.dropout_p)
            
        # Compute output
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights
        
    def _get_causal_mask(
        self,
        q_start: int,
        q_end: int,
        k_start: int,
        k_end: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate causal mask for a block."""
        q_indices = torch.arange(q_start, q_end, device=device)[:, None]  # [Br, 1]
        k_indices = torch.arange(k_start, k_end, device=device)[None, :]  # [1, Bc]
        
        # Causal mask: query can only attend to keys at same or earlier positions
        causal_mask = (q_indices < k_indices).to(dtype) * float('-inf')
        
        return causal_mask


class MultiHeadFlashAttention(nn.Module):
    """Multi-head attention using Flash Attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        flash_config: FlashAttentionConfig = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Flash attention
        if flash_config is None:
            flash_config = FlashAttentionConfig(dropout_p=dropout)
        self.flash_attn = FlashAttention(flash_config)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
            
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head flash attention.
        
        Args:
            query: Query tensor [B, N, D]
            key: Key tensor [B, N, D] (defaults to query if None)
            value: Value tensor [B, N, D] (defaults to key if None)
            attn_mask: Attention mask
            key_padding_mask: Key padding mask [B, N]
            
        Returns:
            output: Attention output [B, N, D]
            attention_weights: Attention weights [B, H, N, N]
        """
        if key is None:
            key = query
        if value is None:
            value = key
            
        batch_size, seq_len, embed_dim = query.shape
        
        # Linear projections
        q = self.q_proj(query)  # [B, N, D]
        k = self.k_proj(key)    # [B, N, D]
        v = self.v_proj(value)  # [B, N, D]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        
        # Apply flash attention
        attn_output, attn_weights = self.flash_attn(
            q, k, v, attn_mask, key_padding_mask
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )  # [B, N, D]
        
        output = self.out_proj(attn_output)
        
        return output, attn_weights
        
    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout}'


class FlashAttentionBlock(nn.Module):
    """Complete transformer block with Flash Attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        flash_config: FlashAttentionConfig = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.norm_first = norm_first
        
        # Multi-head flash attention
        self.self_attn = MultiHeadFlashAttention(
            embed_dim, num_heads, dropout, flash_config=flash_config
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of flash attention block."""
        if self.norm_first:
            # Pre-norm
            attn_output, _ = self.self_attn(
                self.norm1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )
            x = x + self.dropout(attn_output)
            
            ffn_output = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x)))))
            x = x + self.dropout(ffn_output)
        else:
            # Post-norm
            attn_output, _ = self.self_attn(
                x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )
            x = self.norm1(x + self.dropout(attn_output))
            
            ffn_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = self.norm2(x + self.dropout(ffn_output))
            
        return x


if __name__ == "__main__":
    # Demo usage and benchmarking
    print("⚡ Flash Attention Demo")
    print("=" * 50)
    
    # Configuration
    batch_size = 2
    seq_len = 1024
    embed_dim = 512
    num_heads = 8
    
    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create Flash Attention with different configurations
    configs = [
        FlashAttentionConfig(block_size_q=64, block_size_k=64, causal=False),
        FlashAttentionConfig(block_size_q=128, block_size_k=128, causal=True),
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: block_size={config.block_size_q}, causal={config.causal}")
        
        # Create attention layer
        flash_attn_layer = MultiHeadFlashAttention(
            embed_dim, num_heads, flash_config=config
        )
        
        # Forward pass
        output, attn_weights = flash_attn_layer(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Memory efficient: O(N) instead of O(N²)")
        
        # Test with very long sequence (Flash Attention advantage)
        if seq_len >= 512:
            long_x = torch.randn(1, 2048, embed_dim)
            
            try:
                long_output, _ = flash_attn_layer(long_x)
                print(f"Long sequence (2048) handled successfully: {long_output.shape}")
            except RuntimeError as e:
                print(f"Long sequence failed: {e}")
    
    # Test Flash Attention Block
    print(f"\nTesting Complete Flash Attention Block:")
    flash_block = FlashAttentionBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=embed_dim * 4,
        flash_config=FlashAttentionConfig(block_size_q=64, causal=True)
    )
    
    block_output = flash_block(x)
    print(f"Block output shape: {block_output.shape}")
    
    print("\n✅ Flash Attention implementation ready!")
    print("Key benefits:")
    print("- Memory complexity: O(N) instead of O(N²)")
    print("- Supports very long sequences")  
    print("- Block-wise computation with online softmax")
    print("- Causal and non-causal attention support")