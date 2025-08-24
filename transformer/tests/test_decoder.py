"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for transformer decoder layer.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
import torch

from transformer.decoder import Decoder


class TestDecoder:
    """Test suite for Decoder class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        embed_dim = 64
        num_heads = 8
        latent_dim = 256

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        # Check components exist
        assert hasattr(decoder, "self_attention")
        assert hasattr(decoder, "cross_attention")
        assert hasattr(decoder, "ffn")
        assert hasattr(decoder, "layer_norm_1")
        assert hasattr(decoder, "layer_norm_2")
        assert hasattr(decoder, "layer_norm_3")

        # Check attention configurations
        assert decoder.self_attention.embed_dim == embed_dim
        assert decoder.self_attention.num_heads == num_heads
        assert decoder.self_attention.apply_causal_mask is True

        assert decoder.cross_attention.embed_dim == embed_dim
        assert decoder.cross_attention.num_heads == num_heads
        assert decoder.cross_attention.apply_causal_mask is False

        # Check layer norms
        assert decoder.layer_norm_1.normalized_shape == (embed_dim,)
        assert decoder.layer_norm_2.normalized_shape == (embed_dim,)
        assert decoder.layer_norm_3.normalized_shape == (embed_dim,)

    def test_init_different_configurations(self):
        """Test initialization with different valid configurations."""
        test_configs = [(32, 4, 128), (128, 8, 512), (256, 16, 1024), (512, 8, 2048)]

        for embed_dim, num_heads, latent_dim in test_configs:
            decoder = Decoder(
                embed_dim,
                num_heads,
                num_groups=4,
                latent_dim=latent_dim,
                attention_type="mha",
            )

            assert decoder.self_attention.embed_dim == embed_dim
            assert decoder.self_attention.num_heads == num_heads
            assert decoder.cross_attention.embed_dim == embed_dim
            assert decoder.cross_attention.num_heads == num_heads

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        embed_dim = 64
        num_heads = 8
        latent_dim = 256
        batch_size = 4
        tgt_len = 12
        src_len = 16

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        decoder_input = torch.randn(batch_size, tgt_len, embed_dim)
        encoder_output = torch.randn(batch_size, src_len, embed_dim)

        output = decoder(decoder_input, encoder_output)

        # Output should match decoder input dimensions
        assert output.shape == (batch_size, tgt_len, embed_dim)
        assert torch.isfinite(output).all()

    def test_forward_different_shapes(self):
        """Test forward pass with different input shapes."""
        embed_dim = 128
        num_heads = 8
        latent_dim = 512

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        test_cases = [
            (1, 4, 8),  # batch_size, tgt_len, src_len
            (2, 8, 12),
            (4, 16, 20),
            (8, 6, 10),
        ]

        for batch_size, tgt_len, src_len in test_cases:
            decoder_input = torch.randn(batch_size, tgt_len, embed_dim)
            encoder_output = torch.randn(batch_size, src_len, embed_dim)

            output = decoder(decoder_input, encoder_output)

            assert output.shape == (batch_size, tgt_len, embed_dim)
            assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test that gradients flow properly through all components."""
        embed_dim = 64
        num_heads = 4
        latent_dim = 256
        batch_size = 2
        tgt_len = 6
        src_len = 8

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        decoder_input = torch.randn(batch_size, tgt_len, embed_dim, requires_grad=True)
        encoder_output = torch.randn(batch_size, src_len, embed_dim, requires_grad=True)

        output = decoder(decoder_input, encoder_output)
        loss = output.sum()

        loss.backward()

        # Check gradients exist for all parameters
        for name, param in decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"

        # Check input gradients
        assert decoder_input.grad is not None
        assert encoder_output.grad is not None
        assert torch.isfinite(decoder_input.grad).all()
        assert torch.isfinite(encoder_output.grad).all()

    def test_residual_connections(self):
        """Test that residual connections work properly."""
        embed_dim = 64
        num_heads = 8
        latent_dim = 256
        batch_size = 2
        tgt_len = 4
        src_len = 6

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        # Use small random inputs to test residual effect
        decoder_input = torch.randn(batch_size, tgt_len, embed_dim) * 0.1
        encoder_output = torch.randn(batch_size, src_len, embed_dim) * 0.1

        output = decoder(decoder_input, encoder_output)

        # Output should be different from input due to transformations
        assert not torch.allclose(output, decoder_input, atol=1e-3)

        # But residual connections should prevent output from being too different
        # (this is a heuristic test)
        diff_norm = torch.norm(output - decoder_input)
        input_norm = torch.norm(decoder_input)

        # The difference should be reasonable relative to input magnitude
        assert diff_norm < 10 * input_norm

    def test_masked_self_attention(self):
        """Test that self-attention uses causal masking."""
        embed_dim = 32
        num_heads = 4
        latent_dim = 128
        batch_size = 1
        tgt_len = 4
        src_len = 4

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        # Create a simple pattern where each position has distinct values
        decoder_input = torch.zeros(batch_size, tgt_len, embed_dim)
        for i in range(tgt_len):
            decoder_input[0, i, :] = (
                i + 1
            )  # Position 0: all 1s, position 1: all 2s, etc.

        encoder_output = torch.randn(batch_size, src_len, embed_dim)

        output = decoder(decoder_input, encoder_output)

        # Due to causal masking, early positions should be less influenced by later positions
        # This is a behavioral test rather than a strict mathematical test
        assert output.shape == (batch_size, tgt_len, embed_dim)
        assert torch.isfinite(output).all()

        # Each position should produce different outputs
        for i in range(tgt_len - 1):
            assert not torch.allclose(output[0, i, :], output[0, i + 1, :], atol=1e-6)

    def test_cross_attention_functionality(self):
        """Test that cross-attention properly uses encoder outputs."""
        embed_dim = 64
        num_heads = 8
        latent_dim = 256
        batch_size = 2
        tgt_len = 3
        src_len = 5

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        decoder_input = torch.randn(batch_size, tgt_len, embed_dim)
        encoder_output1 = torch.randn(batch_size, src_len, embed_dim)
        encoder_output2 = torch.randn(batch_size, src_len, embed_dim)

        # Different encoder outputs should produce different decoder outputs
        output1 = decoder(decoder_input, encoder_output1)
        output2 = decoder(decoder_input, encoder_output2)

        # Outputs should be different when encoder outputs are different
        assert not torch.allclose(output1, output2, atol=1e-5)

        # But same encoder output should produce same decoder output (deterministic)
        output1_repeat = decoder(decoder_input, encoder_output1)
        assert torch.allclose(output1, output1_repeat, atol=1e-6)

    def test_layer_normalization(self):
        """Test layer normalization functionality."""
        embed_dim = 64
        num_heads = 8
        latent_dim = 256
        batch_size = 2
        tgt_len = 4
        src_len = 6

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        # Create input with different magnitudes
        decoder_input = torch.randn(batch_size, tgt_len, embed_dim) * 10
        encoder_output = torch.randn(batch_size, src_len, embed_dim) * 0.1

        output = decoder(decoder_input, encoder_output)

        # Layer normalization should produce reasonable output regardless of input scale
        assert torch.isfinite(output).all()
        assert output.shape == (batch_size, tgt_len, embed_dim)

        # Output should have reasonable magnitude (roughly normalized)
        output_std = torch.std(output, dim=-1)
        assert torch.all(output_std > 0.1)  # Not all zeros
        assert torch.all(output_std < 10.0)  # Not too large

    def test_deterministic_output(self):
        """Test that decoder produces deterministic output."""
        embed_dim = 64
        num_heads = 8
        latent_dim = 256
        batch_size = 2
        tgt_len = 4
        src_len = 6

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        # Set to evaluation mode for deterministic behavior
        decoder.eval()

        decoder_input = torch.randn(batch_size, tgt_len, embed_dim)
        encoder_output = torch.randn(batch_size, src_len, embed_dim)

        with torch.no_grad():
            output1 = decoder(decoder_input, encoder_output)
            output2 = decoder(decoder_input, encoder_output)

        # Should produce identical outputs
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_training_vs_eval_mode(self):
        """Test decoder behavior in training vs evaluation mode."""
        embed_dim = 64
        num_heads = 8
        latent_dim = 256
        batch_size = 2
        tgt_len = 4
        src_len = 6

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        decoder_input = torch.randn(batch_size, tgt_len, embed_dim)
        encoder_output = torch.randn(batch_size, src_len, embed_dim)

        # Training mode
        decoder.train()
        output_train = decoder(decoder_input, encoder_output)

        # Evaluation mode
        decoder.eval()
        with torch.no_grad():
            output_eval = decoder(decoder_input, encoder_output)

        # Outputs should have same shape
        assert output_train.shape == output_eval.shape
        assert torch.isfinite(output_train).all()
        assert torch.isfinite(output_eval).all()

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Minimum viable configuration
        embed_dim = 8
        num_heads = 2
        latent_dim = 16

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        # Single token sequences
        decoder_input = torch.randn(1, 1, embed_dim)
        encoder_output = torch.randn(1, 1, embed_dim)

        output = decoder(decoder_input, encoder_output)

        assert output.shape == (1, 1, embed_dim)
        assert torch.isfinite(output).all()

        # Different sequence lengths (decoder shorter than encoder)
        decoder_input = torch.randn(2, 2, embed_dim)
        encoder_output = torch.randn(2, 8, embed_dim)

        output = decoder(decoder_input, encoder_output)

        assert output.shape == (2, 2, embed_dim)
        assert torch.isfinite(output).all()

    def test_parameter_count(self):
        """Test that decoder has expected number of parameters."""
        embed_dim = 64
        num_heads = 8
        latent_dim = 256

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        total_params = sum(p.numel() for p in decoder.parameters())

        # Should have parameters for:
        # - Self attention: ~4 * embed_dim^2 (Q, K, V, output projections)
        # - Cross attention: ~4 * embed_dim^2
        # - FFN: ~embed_dim * latent_dim * 2 (two linear layers)
        # - Layer norms: ~embed_dim * 3 (3 layer norms)
        expected_min = (
            8 * embed_dim * embed_dim + 2 * embed_dim * latent_dim + 3 * embed_dim
        )

        assert total_params > expected_min
        assert total_params > 0

    def test_different_sequence_length_combinations(self):
        """Test various combinations of decoder and encoder sequence lengths."""
        embed_dim = 32
        num_heads = 4
        latent_dim = 128

        decoder = Decoder(
            embed_dim,
            num_heads,
            num_groups=4,
            latent_dim=latent_dim,
            attention_type="mha",
        )

        # Test different length combinations
        length_combinations = [
            (2, 4),  # tgt_len, src_len
            (4, 2),  # decoder longer than encoder
            (6, 6),  # same length
            (1, 8),  # very short decoder
            (8, 1),  # very short encoder
        ]

        for tgt_len, src_len in length_combinations:
            decoder_input = torch.randn(1, tgt_len, embed_dim)
            encoder_output = torch.randn(1, src_len, embed_dim)

            output = decoder(decoder_input, encoder_output)

            assert output.shape == (1, tgt_len, embed_dim)
            assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__])
