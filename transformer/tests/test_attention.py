"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for multi-head attention mechanism.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch

from transformer.attention.mha import MultiHeadAttention


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        embed_dim = 64
        num_heads = 8
        attn = MultiHeadAttention(embed_dim, num_heads)

        assert attn.embed_dim == embed_dim
        assert attn.num_heads == num_heads
        assert attn.head_dim == embed_dim // num_heads
        assert attn.sqrt_d == (embed_dim // num_heads) ** 0.5

    def test_init_invalid_params(self):
        """Test initialization fails with invalid parameters."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(embed_dim=65, num_heads=8)  # 65 not divisible by 8

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        embed_dim = 64
        num_heads = 8
        batch_size = 4
        seq_len = 16

        attn = MultiHeadAttention(embed_dim, num_heads)
        x = torch.randn(batch_size, seq_len, embed_dim)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_forward_gradient_flow(self):
        """Test gradients flow through attention mechanism."""
        embed_dim = 32
        num_heads = 4
        batch_size = 2
        seq_len = 8

        attn = MultiHeadAttention(embed_dim, num_heads)
        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_attention_weights_range(self):
        """Test attention mechanism produces reasonable outputs."""
        embed_dim = 32
        num_heads = 4
        batch_size = 2
        seq_len = 8

        attn = MultiHeadAttention(embed_dim, num_heads)
        x = torch.randn(batch_size, seq_len, embed_dim)

        output = attn(x)

        # Output should be finite
        assert torch.isfinite(output).all()

        # Output should have reasonable magnitude
        assert output.abs().max() < 100.0

    def test_different_input_sizes(self):
        """Test attention works with different sequence lengths."""
        embed_dim = 64
        num_heads = 8
        batch_size = 3

        attn = MultiHeadAttention(embed_dim, num_heads)

        for seq_len in [1, 10, 50]:
            x = torch.randn(batch_size, seq_len, embed_dim)
            output = attn(x)
            assert output.shape == (batch_size, seq_len, embed_dim)

    def test_self_attention_properties(self):
        """Test self-attention mathematical properties."""
        embed_dim = 32
        num_heads = 4
        batch_size = 2
        seq_len = 4

        attn = MultiHeadAttention(embed_dim, num_heads)

        # Test with identical inputs - should produce consistent outputs
        x = torch.ones(batch_size, seq_len, embed_dim)
        output = attn(x)

        # All positions should receive similar attention to identical inputs
        variance_across_positions = output.var(dim=1).mean()
        assert variance_across_positions < 1.0  # Low variance expected

    def test_zero_input(self):
        """Test attention handles zero input gracefully."""
        embed_dim = 32
        num_heads = 4
        batch_size = 2
        seq_len = 8

        attn = MultiHeadAttention(embed_dim, num_heads)
        x = torch.zeros(batch_size, seq_len, embed_dim)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__])
