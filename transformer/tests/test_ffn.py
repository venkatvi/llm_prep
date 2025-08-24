"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for feedforward network.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
import torch

from transformer.ffn import FFN


class TestFFN:
    """Test suite for FFN class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        embed_dim = 64
        latent_dim = 256
        ffn = FFN(embed_dim, latent_dim)

        assert ffn.layer_1.in_features == embed_dim
        assert ffn.layer_1.out_features == latent_dim
        assert ffn.layer_2.in_features == latent_dim
        assert ffn.layer_2.out_features == embed_dim
        assert isinstance(ffn.relu, torch.nn.ReLU)

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        embed_dim = 64
        latent_dim = 256
        batch_size = 4
        seq_len = 16

        ffn = FFN(embed_dim, latent_dim)
        x = torch.randn(batch_size, seq_len, embed_dim)

        output = ffn(x)

        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_forward_gradient_flow(self):
        """Test gradients flow through FFN."""
        embed_dim = 32
        latent_dim = 128
        batch_size = 2
        seq_len = 8

        ffn = FFN(embed_dim, latent_dim)
        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

        output = ffn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_relu_activation(self):
        """Test ReLU activation works correctly."""
        embed_dim = 8
        latent_dim = 16
        batch_size = 2
        seq_len = 4

        ffn = FFN(embed_dim, latent_dim)

        # Create input that will produce negative values in latent layer
        x = torch.randn(batch_size, seq_len, embed_dim)

        output = ffn(x)

        # Output should be finite
        assert torch.isfinite(output).all()

    def test_different_dimensions(self):
        """Test FFN works with different embed/latent dimensions."""
        test_cases = [(16, 64), (32, 128), (64, 256), (128, 512)]

        batch_size = 2
        seq_len = 8

        for embed_dim, latent_dim in test_cases:
            ffn = FFN(embed_dim, latent_dim)
            x = torch.randn(batch_size, seq_len, embed_dim)
            output = ffn(x)
            assert output.shape == (batch_size, seq_len, embed_dim)

    def test_zero_input(self):
        """Test FFN handles zero input gracefully."""
        embed_dim = 32
        latent_dim = 128
        batch_size = 2
        seq_len = 8

        ffn = FFN(embed_dim, latent_dim)
        x = torch.zeros(batch_size, seq_len, embed_dim)

        output = ffn(x)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert torch.isfinite(output).all()

    def test_positive_input_positive_output(self):
        """Test that positive inputs can produce positive outputs."""
        embed_dim = 32
        latent_dim = 128
        batch_size = 2
        seq_len = 8

        ffn = FFN(embed_dim, latent_dim)

        # Use positive input
        x = torch.abs(torch.randn(batch_size, seq_len, embed_dim))
        output = ffn(x)

        # Should have some positive values in output
        assert (output > 0).any()

    def test_linear_layer_properties(self):
        """Test linear layer properties."""
        embed_dim = 32
        latent_dim = 128

        ffn = FFN(embed_dim, latent_dim)

        # Check weight and bias existence
        assert ffn.layer_1.weight.shape == (latent_dim, embed_dim)
        assert ffn.layer_1.bias.shape == (latent_dim,)
        assert ffn.layer_2.weight.shape == (embed_dim, latent_dim)
        assert ffn.layer_2.bias.shape == (embed_dim,)

    def test_network_depth(self):
        """Test that network applies transformations in correct order."""
        embed_dim = 4
        latent_dim = 8

        ffn = FFN(embed_dim, latent_dim)

        # Manual forward pass to check order
        x = torch.randn(1, 1, embed_dim)

        # Step by step
        layer1_out = ffn.layer_1(x)
        assert layer1_out.shape == (1, 1, latent_dim)

        relu_out = ffn.relu(layer1_out)
        assert relu_out.shape == (1, 1, latent_dim)
        assert (relu_out >= 0).all()  # ReLU ensures non-negative

        layer2_out = ffn.layer_2(relu_out)
        assert layer2_out.shape == (1, 1, embed_dim)

        # Compare with direct forward
        direct_out = ffn(x)
        assert torch.allclose(layer2_out, direct_out)


if __name__ == "__main__":
    pytest.main([__file__])
