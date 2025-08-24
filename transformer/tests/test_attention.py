"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for multi-head attention mechanism.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
import torch

from transformer.attention.mha import MultiHeadAttention
from transformer.attention.mqa import MultiQueryAttention
from transformer.attention.gqa import GroupQueryAttention


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        embed_dim = 64
        num_heads = 8
        attn = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask=False)

        assert attn.embed_dim == embed_dim
        assert attn.num_heads == num_heads
        assert attn.head_dim == embed_dim // num_heads
        assert attn.sqrt_d == (embed_dim // num_heads) ** 0.5

    def test_init_invalid_params(self):
        """Test initialization fails with invalid parameters."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(
                embed_dim=65, num_heads=8, apply_causal_mask=False
            )  # 65 not divisible by 8

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        embed_dim = 64
        num_heads = 8
        batch_size = 4
        seq_len = 16

        attn = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask=False)
        x = torch.randn(batch_size, seq_len, embed_dim)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_forward_gradient_flow(self):
        """Test gradients flow through attention mechanism."""
        embed_dim = 32
        num_heads = 4
        batch_size = 2
        seq_len = 8

        attn = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask=False)
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

        attn = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask=False)
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

        attn = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask=False)

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

        attn = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask=False)

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

        attn = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask=False)
        x = torch.zeros(batch_size, seq_len, embed_dim)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert torch.isfinite(output).all()


class TestMultiQueryAttention:
    """Test suite for MultiQueryAttention class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        embed_dim = 64
        num_heads = 8
        attn = MultiQueryAttention(embed_dim, num_heads, apply_causal_mask=False)

        assert attn.embed_dim == embed_dim
        assert attn.num_heads == num_heads
        assert attn.head_dim == embed_dim // num_heads

    def test_init_invalid_params(self):
        """Test initialization fails with invalid parameters."""
        with pytest.raises(AssertionError):
            MultiQueryAttention(
                embed_dim=65, num_heads=8, apply_causal_mask=False
            )  # 65 not divisible by 8

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        embed_dim = 64
        num_heads = 8
        batch_size = 4
        seq_len = 16

        attn = MultiQueryAttention(embed_dim, num_heads, apply_causal_mask=False)
        x = torch.randn(batch_size, seq_len, embed_dim)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test gradients flow properly through MQA."""
        embed_dim = 32
        num_heads = 4
        batch_size = 2
        seq_len = 8

        attn = MultiQueryAttention(embed_dim, num_heads, apply_causal_mask=False)
        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check all parameters have gradients
        for param in attn.parameters():
            assert param.grad is not None

    def test_different_configurations(self):
        """Test MQA with different head configurations."""
        configs = [(32, 4), (64, 8), (128, 16)]

        batch_size = 2
        seq_len = 10

        for embed_dim, num_heads in configs:
            attn = MultiQueryAttention(embed_dim, num_heads, apply_causal_mask=False)
            x = torch.randn(batch_size, seq_len, embed_dim)

            output = attn(x)
            assert output.shape == (batch_size, seq_len, embed_dim)
            assert torch.isfinite(output).all()


class TestGroupQueryAttention:
    """Test suite for GroupQueryAttention class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        embed_dim = 64
        num_heads = 8
        num_groups = 4
        attn = GroupQueryAttention(
            embed_dim, num_heads, num_groups, apply_causal_mask=False
        )

        assert attn.embed_dim == embed_dim
        assert attn.num_heads == num_heads
        assert attn.num_groups == num_groups
        assert attn.head_dim == embed_dim // num_heads
        assert attn.group_size == num_heads // num_groups

    def test_init_invalid_params(self):
        """Test initialization fails with invalid parameters."""
        # embed_dim not divisible by num_heads
        with pytest.raises(AssertionError):
            GroupQueryAttention(
                embed_dim=65, num_heads=8, num_groups=4, apply_causal_mask=False
            )

        # num_heads not divisible by num_groups
        with pytest.raises(AssertionError):
            GroupQueryAttention(
                embed_dim=64, num_heads=9, num_groups=4, apply_causal_mask=False
            )

        # num_groups >= num_heads
        with pytest.raises(AssertionError):
            GroupQueryAttention(
                embed_dim=64, num_heads=8, num_groups=8, apply_causal_mask=False
            )

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        embed_dim = 64
        num_heads = 8
        num_groups = 4
        batch_size = 4
        seq_len = 16

        attn = GroupQueryAttention(
            embed_dim, num_heads, num_groups, apply_causal_mask=False
        )
        x = torch.randn(batch_size, seq_len, embed_dim)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test gradients flow properly through GQA."""
        embed_dim = 32
        num_heads = 8
        num_groups = 4
        batch_size = 2
        seq_len = 8

        attn = GroupQueryAttention(
            embed_dim, num_heads, num_groups, apply_causal_mask=False
        )
        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check all parameters have gradients
        for param in attn.parameters():
            assert param.grad is not None

    def test_different_group_configurations(self):
        """Test GQA with different group configurations."""
        configs = [
            (32, 8, 2),  # 4 heads per group
            (64, 8, 4),  # 2 heads per group
            (128, 16, 8),  # 2 heads per group
        ]

        batch_size = 2
        seq_len = 10

        for embed_dim, num_heads, num_groups in configs:
            attn = GroupQueryAttention(
                embed_dim, num_heads, num_groups, apply_causal_mask=False
            )
            x = torch.randn(batch_size, seq_len, embed_dim)

            output = attn(x)
            assert output.shape == (batch_size, seq_len, embed_dim)
            assert torch.isfinite(output).all()


class TestAttentionComparison:
    """Test comparison between different attention mechanisms."""

    def test_output_shapes_consistent(self):
        """Test all attention mechanisms produce same output shape."""
        embed_dim = 64
        num_heads = 8
        num_groups = 4
        batch_size = 2
        seq_len = 12

        x = torch.randn(batch_size, seq_len, embed_dim)

        mha = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask=False)
        mqa = MultiQueryAttention(embed_dim, num_heads, apply_causal_mask=False)
        gqa = GroupQueryAttention(
            embed_dim, num_heads, num_groups, apply_causal_mask=False
        )

        mha_out = mha(x)
        mqa_out = mqa(x)
        gqa_out = gqa(x)

        expected_shape = (batch_size, seq_len, embed_dim)
        assert mha_out.shape == expected_shape
        assert mqa_out.shape == expected_shape
        assert gqa_out.shape == expected_shape

    def test_parameter_count_differences(self):
        """Test parameter count differences between attention types."""
        embed_dim = 64
        num_heads = 8
        num_groups = 4

        mha = MultiHeadAttention(embed_dim, num_heads, apply_causal_mask=False)
        mqa = MultiQueryAttention(embed_dim, num_heads, apply_causal_mask=False)
        gqa = GroupQueryAttention(
            embed_dim, num_heads, num_groups, apply_causal_mask=False
        )

        mha_params = sum(p.numel() for p in mha.parameters())
        mqa_params = sum(p.numel() for p in mqa.parameters())
        gqa_params = sum(p.numel() for p in gqa.parameters())

        # MQA should have fewer parameters than MHA (single K,V heads)
        assert mqa_params < mha_params

        # GQA should be between MHA and MQA
        assert gqa_params < mha_params
        assert gqa_params > mqa_params


if __name__ == "__main__":
    pytest.main([__file__])
