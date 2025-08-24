"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for positional encoding.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import math

import pytest
import torch

from transformer.input_encodings import PositionalEncoding


class TestPositionalEncoding:
    """Test suite for PositionalEncoding class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        seq_len = 100
        d_model = 64
        pe = PositionalEncoding(seq_len, d_model)

        assert pe.seq_len == seq_len
        assert pe.d_model == d_model
        assert pe.pe.shape == (1, seq_len, d_model)

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        seq_len = 100
        d_model = 64
        batch_size = 4
        input_seq_len = 32

        pe = PositionalEncoding(seq_len, d_model)
        x = torch.randn(batch_size, input_seq_len, d_model)

        output = pe(x)

        assert output.shape == (batch_size, input_seq_len, d_model)

    def test_forward_gradient_flow(self):
        """Test gradients flow through positional encoding."""
        seq_len = 50
        d_model = 32
        batch_size = 2
        input_seq_len = 16

        pe = PositionalEncoding(seq_len, d_model)
        x = torch.randn(batch_size, input_seq_len, d_model, requires_grad=True)

        output = pe(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_positional_encoding_pattern(self):
        """Test positional encoding follows sinusoidal pattern."""
        seq_len = 10
        d_model = 8

        pe = PositionalEncoding(seq_len, d_model)
        pos_encoding = pe.pe[0]  # Remove batch dimension

        # Check even indices use sine
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                expected_angle = pos / (10000 ** (2 * (i // 2) / d_model))
                expected_value = math.sin(expected_angle)
                actual_value = pos_encoding[pos, i].item()
                assert abs(actual_value - expected_value) < 1e-5

    def test_positional_encoding_odd_pattern(self):
        """Test positional encoding odd indices use cosine."""
        seq_len = 10
        d_model = 8

        pe = PositionalEncoding(seq_len, d_model)
        pos_encoding = pe.pe[0]  # Remove batch dimension

        # Check odd indices use cosine
        for pos in range(seq_len):
            for i in range(1, d_model, 2):
                expected_angle = pos / (10000 ** (2 * (i // 2) / d_model))
                expected_value = math.cos(expected_angle)
                actual_value = pos_encoding[pos, i].item()
                assert abs(actual_value - expected_value) < 1e-5

    def test_different_sequence_lengths(self):
        """Test encoding works with different input sequence lengths."""
        max_seq_len = 100
        d_model = 32
        batch_size = 2

        pe = PositionalEncoding(max_seq_len, d_model)

        for input_seq_len in [1, 10, 50, 100]:
            x = torch.randn(batch_size, input_seq_len, d_model)
            output = pe(x)
            assert output.shape == (batch_size, input_seq_len, d_model)

    def test_sequence_length_exceeds_max(self):
        """Test behavior when input sequence exceeds max length."""
        max_seq_len = 10
        d_model = 32
        batch_size = 2
        input_seq_len = 15  # Exceeds max_seq_len

        pe = PositionalEncoding(max_seq_len, d_model)
        x = torch.randn(batch_size, input_seq_len, d_model)

        # Should handle gracefully by truncating positional encoding
        output = pe(x)
        assert output.shape == (batch_size, input_seq_len, d_model)

    def test_zero_input(self):
        """Test positional encoding with zero input."""
        seq_len = 20
        d_model = 16
        batch_size = 2
        input_seq_len = 10

        pe = PositionalEncoding(seq_len, d_model)
        x = torch.zeros(batch_size, input_seq_len, d_model)

        output = pe(x)

        # Output should equal positional encoding since input is zero
        expected = pe.pe[:, :input_seq_len, :].expand(batch_size, -1, -1)
        assert torch.allclose(output, expected)

    def test_encoding_uniqueness(self):
        """Test that different positions have different encodings."""
        seq_len = 20
        d_model = 32

        pe = PositionalEncoding(seq_len, d_model)
        pos_encoding = pe.pe[0]  # Remove batch dimension

        # Check that different positions have different encodings
        for i in range(seq_len - 1):
            for j in range(i + 1, seq_len):
                assert not torch.allclose(pos_encoding[i], pos_encoding[j])

    def test_encoding_magnitude(self):
        """Test positional encoding values are in reasonable range."""
        seq_len = 100
        d_model = 64

        pe = PositionalEncoding(seq_len, d_model)
        pos_encoding = pe.pe[0]

        # Values should be between -1 and 1 (sin/cos range)
        assert (pos_encoding >= -1.0).all()
        assert (pos_encoding <= 1.0).all()

    def test_even_odd_dimension_handling(self):
        """Test encoding works with both even and odd dimensions."""
        seq_len = 10

        for d_model in [16, 17]:  # Even and odd dimensions
            pe = PositionalEncoding(seq_len, d_model)
            x = torch.randn(2, 5, d_model)
            output = pe(x)
            assert output.shape == (2, 5, d_model)
            assert torch.isfinite(output).all()

    def test_buffer_registration(self):
        """Test that positional encoding is properly registered as buffer."""
        seq_len = 20
        d_model = 32

        pe = PositionalEncoding(seq_len, d_model)

        # Check buffer is registered
        assert "pe" in pe._buffers
        assert pe._buffers["pe"] is pe.pe


if __name__ == "__main__":
    pytest.main([__file__])
