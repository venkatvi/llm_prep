"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for regression configuration classes.
"""

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from regression.configs import AutoregressiveDecodeConfig, RegressionModelConfig, TransformerModelConfig


class TestAutoregressiveDecodeConfig(unittest.TestCase):
    """Test suite for AutoregressiveDecodeConfig class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        config = AutoregressiveDecodeConfig(num_steps=10, expanding_context=True, max_seq_len=50)

        self.assertEqual(config.num_steps, 10)
        self.assertTrue(config.expanding_context)
        self.assertEqual(config.max_seq_len, 50)

    def test_init_different_values(self):
        """Test initialization with different valid values."""
        config = AutoregressiveDecodeConfig(num_steps=5, expanding_context=False, max_seq_len=100)

        self.assertEqual(config.num_steps, 5)
        self.assertFalse(config.expanding_context)
        self.assertEqual(config.max_seq_len, 100)


class TestRegressionModelConfig(unittest.TestCase):
    """Test suite for RegressionModelConfig class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        config = RegressionModelConfig(
            name="test_model", custom_act="relu", num_latent_layers=3, latent_dims=[64, 32, 16], allow_residual=True
        )

        self.assertEqual(config.name, "test_model")
        self.assertEqual(config.custom_act, "relu")
        self.assertEqual(config.num_latent_layers, 3)
        self.assertEqual(config.latent_dims, [64, 32, 16])
        self.assertTrue(config.allow_residual)

    def test_different_activations(self):
        """Test with different activation functions."""
        activations = ["relu", "tanh", "sigmoid", "gelu"]

        for act in activations:
            with self.subTest(activation=act):
                config = RegressionModelConfig(
                    name="test", custom_act=act, num_latent_layers=2, latent_dims=[32, 16], allow_residual=False
                )
                self.assertEqual(config.custom_act, act)

    def test_various_layer_configurations(self):
        """Test with different layer configurations."""
        test_cases = [(1, [32]), (2, [64, 32]), (3, [128, 64, 32]), (4, [256, 128, 64, 32])]

        for num_layers, dims in test_cases:
            with self.subTest(layers=num_layers, dims=dims):
                config = RegressionModelConfig(
                    name="test", custom_act="relu", num_latent_layers=num_layers, latent_dims=dims, allow_residual=False
                )
                self.assertEqual(config.num_latent_layers, num_layers)
                self.assertEqual(config.latent_dims, dims)

    def test_residual_connection_options(self):
        """Test residual connection enable/disable."""
        for residual in [True, False]:
            with self.subTest(residual=residual):
                config = RegressionModelConfig(
                    name="test", custom_act="relu", num_latent_layers=2, latent_dims=[32, 16], allow_residual=residual
                )
                self.assertEqual(config.allow_residual, residual)


class TestTransformerModelConfig(unittest.TestCase):
    """Test suite for TransformerModelConfig class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        decode_config = AutoregressiveDecodeConfig(num_steps=10, expanding_context=True, max_seq_len=50)

        config = TransformerModelConfig(
            name="transformer_test",
            max_seq_len=64,
            input_dim=8,
            embed_dim=128,
            ffn_latent_dim=512,
            num_layers=6,
            output_dim=1,
            num_heads=8,
            apply_causal_mask=True,
            autoregressive_mode=True,
            decode_config=decode_config,
            attention_type="mha",
        )

        self.assertEqual(config.name, "transformer_test")
        self.assertEqual(config.max_seq_len, 64)
        self.assertEqual(config.input_dim, 8)
        self.assertEqual(config.embed_dim, 128)
        self.assertEqual(config.ffn_latent_dim, 512)
        self.assertEqual(config.num_layers, 6)
        self.assertEqual(config.output_dim, 1)
        self.assertEqual(config.num_heads, 8)
        self.assertTrue(config.apply_causal_mask)
        self.assertTrue(config.autoregressive_mode)
        self.assertEqual(config.decode_config, decode_config)

    def test_different_dimensions(self):
        """Test with different model dimensions."""
        decode_config = AutoregressiveDecodeConfig(num_steps=5, expanding_context=False, max_seq_len=32)

        test_cases = [
            {"embed_dim": 64, "ffn_latent_dim": 256, "num_heads": 4},
            {"embed_dim": 256, "ffn_latent_dim": 1024, "num_heads": 16},
            {"embed_dim": 512, "ffn_latent_dim": 2048, "num_heads": 8},
        ]

        for params in test_cases:
            with self.subTest(**params):
                config = TransformerModelConfig(
                    name="test",
                    max_seq_len=32,
                    input_dim=4,
                    embed_dim=params["embed_dim"],
                    ffn_latent_dim=params["ffn_latent_dim"],
                    num_layers=3,
                    output_dim=1,
                    num_heads=params["num_heads"],
                    apply_causal_mask=False,
                    autoregressive_mode=False,
                    decode_config=decode_config,
                    attention_type="mha",
                )
                self.assertEqual(config.embed_dim, params["embed_dim"])
                self.assertEqual(config.ffn_latent_dim, params["ffn_latent_dim"])
                self.assertEqual(config.num_heads, params["num_heads"])

    def test_causal_mask_options(self):
        """Test causal masking enable/disable."""
        decode_config = AutoregressiveDecodeConfig(num_steps=5, expanding_context=True, max_seq_len=32)

        for causal in [True, False]:
            with self.subTest(causal=causal):
                config = TransformerModelConfig(
                    name="test",
                    max_seq_len=32,
                    input_dim=4,
                    embed_dim=64,
                    ffn_latent_dim=256,
                    num_layers=2,
                    output_dim=1,
                    num_heads=4,
                    apply_causal_mask=causal,
                    autoregressive_mode=causal,
                    decode_config=decode_config,
                    attention_type="mha",
                )
                self.assertEqual(config.apply_causal_mask, causal)
                self.assertEqual(config.autoregressive_mode, causal)

    def test_different_attention_types(self):
        """Test with different attention mechanisms."""
        decode_config = AutoregressiveDecodeConfig(num_steps=5, expanding_context=True, max_seq_len=32)

        attention_types = ["mha", "mqa", "gqa"]

        for att_type in attention_types:
            with self.subTest(attention_type=att_type):
                config = TransformerModelConfig(
                    name="test",
                    max_seq_len=32,
                    input_dim=4,
                    embed_dim=64,
                    ffn_latent_dim=256,
                    num_layers=2,
                    output_dim=1,
                    num_heads=4,
                    apply_causal_mask=False,
                    autoregressive_mode=False,
                    decode_config=decode_config,
                    attention_type=att_type,
                )
                self.assertEqual(config.attention_type, att_type)


if __name__ == "__main__":
    unittest.main()
