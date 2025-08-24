"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for regression model implementations.
"""

import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from regression.configs import RegressionModelConfig
from regression.e_linear_reg import LinearRegressionModel
from regression.e_non_linear_reg import MLP
from regression.h_transformer import ARTransformerModel, EncoderDecoderWrapper, RegressionTransformerModel


class TestLinearRegressionModel(unittest.TestCase):
    """Test suite for LinearRegressionModel class."""

    def test_init_basic(self):
        """Test basic initialization."""
        config = RegressionModelConfig(
            name="linear_test", custom_act="relu", num_latent_layers=1, latent_dims=[32], allow_residual=False
        )

        model = LinearRegressionModel(config)

        # Check model structure
        self.assertIsInstance(model.linear, torch.nn.Linear)
        self.assertEqual(model.linear.in_features, 1)
        self.assertEqual(model.linear.out_features, 1)

    def test_forward_pass(self):
        """Test forward pass with different input sizes."""
        config = RegressionModelConfig(
            name="linear_test", custom_act="relu", num_latent_layers=1, latent_dims=[32], allow_residual=False
        )

        model = LinearRegressionModel(config)

        # Test different batch sizes
        test_sizes = [(1, 1), (5, 1), (10, 1), (32, 1)]

        for batch_size, input_dim in test_sizes:
            with self.subTest(batch_size=batch_size, input_dim=input_dim):
                x = torch.randn(batch_size, input_dim)
                output = model(x)

                self.assertEqual(output.shape, (batch_size, 1))
                self.assertTrue(torch.isfinite(output).all())

    def test_different_activations(self):
        """Test model with different activation functions."""
        activations = ["relu", "tanh", "sigmoid"]

        for act in activations:
            with self.subTest(activation=act):
                config = RegressionModelConfig(
                    name="test", custom_act=act, num_latent_layers=1, latent_dims=[16], allow_residual=False
                )

                model = LinearRegressionModel(config)
                x = torch.randn(5, 1)
                output = model(x)

                self.assertEqual(output.shape, (5, 1))
                self.assertTrue(torch.isfinite(output).all())

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        config = RegressionModelConfig(
            name="test", custom_act="relu", num_latent_layers=1, latent_dims=[16], allow_residual=False
        )

        model = LinearRegressionModel(config)
        x = torch.randn(3, 1, requires_grad=True)
        output = model(x)
        loss = output.sum()

        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestMLP(unittest.TestCase):
    """Test suite for MLP class."""

    def test_init_single_layer(self):
        """Test initialization with single hidden layer."""
        config = RegressionModelConfig(
            name="mlp_test", custom_act="relu", num_latent_layers=1, latent_dims=[64], allow_residual=False
        )

        model = MLP(config)

        # Check layer structure
        self.assertEqual(len(model.layers), 2)  # Linear + Activation
        self.assertIsInstance(model.output_layer, torch.nn.Linear)
        self.assertEqual(model.output_layer.out_features, 1)

    def test_init_multiple_layers(self):
        """Test initialization with multiple hidden layers."""
        config = RegressionModelConfig(
            name="mlp_test", custom_act="tanh", num_latent_layers=3, latent_dims=[64, 32, 16], allow_residual=False
        )

        model = MLP(config)

        # Should have 6 layers: 3 * (Linear + Activation)
        self.assertEqual(len(model.layers), 6)
        self.assertEqual(model.output_layer.out_features, 1)

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct output shapes."""
        config = RegressionModelConfig(
            name="mlp_test", custom_act="relu", num_latent_layers=2, latent_dims=[32, 16], allow_residual=False
        )

        model = MLP(config)

        # Test different input shapes
        test_cases = [(1, 1), (8, 1), (16, 1), (64, 1)]

        for batch_size, input_dim in test_cases:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, input_dim)
                output = model(x)

                self.assertEqual(output.shape, (batch_size, 1))
                self.assertTrue(torch.isfinite(output).all())

    def test_residual_connections(self):
        """Test model with residual connections enabled."""
        config = RegressionModelConfig(
            name="mlp_test",
            custom_act="relu",
            num_latent_layers=2,
            latent_dims=[16, 16],  # Same size for residual connections
            allow_residual=True,
        )

        model = MLP(config)
        x = torch.randn(5, 1)
        output = model(x)

        self.assertEqual(output.shape, (5, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_different_layer_sizes(self):
        """Test with various layer configurations."""
        test_configs = [([32], 1), ([64, 32], 2), ([128, 64, 32], 3), ([256, 128, 64, 32], 4)]

        for latent_dims, num_layers in test_configs:
            with self.subTest(layers=latent_dims):
                config = RegressionModelConfig(
                    name="test",
                    custom_act="relu",
                    num_latent_layers=num_layers,
                    latent_dims=latent_dims,
                    allow_residual=False,
                )

                model = MLP(config)
                x = torch.randn(4, 1)
                output = model(x)

                self.assertEqual(output.shape, (4, 1))
                self.assertTrue(torch.isfinite(output).all())

    def test_gradient_flow_mlp(self):
        """Test gradient flow through MLP layers."""
        config = RegressionModelConfig(
            name="test", custom_act="relu", num_latent_layers=2, latent_dims=[32, 16], allow_residual=False
        )

        model = MLP(config)
        x = torch.randn(3, 1, requires_grad=True)
        output = model(x)
        loss = output.mean()

        loss.backward()

        # Check all parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())


class TestTransformerModels(unittest.TestCase):
    """Test suite for transformer-based regression models."""

    def test_regression_transformer_model_init(self):
        """Test RegressionTransformerModel initialization."""
        try:
            from regression.configs import TransformerModelConfig, AutoregressiveDecodeConfig

            decode_config = AutoregressiveDecodeConfig(num_steps=5, expanding_context=True, max_seq_len=32)

            config = TransformerModelConfig(
                name="transformer_test",
                max_seq_len=16,
                input_dim=4,
                embed_dim=64,
                ffn_latent_dim=256,
                num_layers=2,
                output_dim=1,
                num_heads=4,
                apply_causal_mask=False,
                autoregressive_mode=False,
                decode_config=decode_config,
                attention_type="mha",
            )

            model = RegressionTransformerModel(config)

            # Test forward pass
            x = torch.randn(2, 16, 4)  # [batch, seq_len, input_dim]
            output = model(x)

            self.assertEqual(output.shape, (2, 1))  # Global average pooling to scalar
            self.assertTrue(torch.isfinite(output).all())

        except ImportError:
            self.skipTest("Transformer dependencies not available")

    def test_ar_transformer_model_basic(self):
        """Test ARTransformerModel basic functionality."""
        try:
            from regression.configs import TransformerModelConfig, AutoregressiveDecodeConfig

            decode_config = AutoregressiveDecodeConfig(num_steps=3, expanding_context=True, max_seq_len=16)

            config = TransformerModelConfig(
                name="ar_test",
                max_seq_len=8,
                input_dim=2,
                embed_dim=32,
                ffn_latent_dim=128,
                num_layers=1,
                output_dim=2,
                num_heads=2,
                apply_causal_mask=True,
                autoregressive_mode=True,
                decode_config=decode_config,
                attention_type="mha",
            )

            model = ARTransformerModel(config)

            # Test forward pass
            x = torch.randn(1, 4, 2)  # [batch, seq_len, input_dim]
            output = model(x)

            self.assertEqual(output.shape, (1, 4, 2))  # Same shape as input
            self.assertTrue(torch.isfinite(output).all())

        except ImportError:
            self.skipTest("Transformer dependencies not available")

    def test_encoder_decoder_wrapper(self):
        """Test EncoderDecoderWrapper functionality."""
        try:
            from regression.configs import EncoderDecoderConfig, AutoregressiveDecodeConfig

            decode_config = AutoregressiveDecodeConfig(num_steps=5, expanding_context=True, max_seq_len=16)

            config = EncoderDecoderConfig(
                name="ed_test",
                max_seq_len=8,
                input_dim=3,
                embed_dim=32,
                ffn_latent_dim=128,
                num_encoder_layers=2,
                num_decoder_layers=2,
                output_dim=3,
                num_heads=4,
                apply_causal_mask=True,
                autoregressive_mode=True,
                decode_config=decode_config,
                attention_type="mha",
            )

            model = EncoderDecoderWrapper(config)

            # Test encode
            source = torch.randn(1, 8, 3)  # [batch, source_len, input_dim]
            encoder_output = model.encode(source)

            self.assertEqual(encoder_output.shape, (1, 8, 32))  # [batch, source_len, embed_dim]

            # Test decode
            target = torch.randn(1, 6, 3)  # [batch, target_len, input_dim]
            decoder_output = model.decode(target, encoder_output)

            self.assertEqual(decoder_output.shape, (1, 6, 3))  # [batch, target_len, input_dim]
            self.assertTrue(torch.isfinite(decoder_output).all())

        except ImportError:
            self.skipTest("Transformer dependencies not available")

    def test_model_training_mode(self):
        """Test models work correctly in training vs eval mode."""
        config = RegressionModelConfig(
            name="test", custom_act="relu", num_latent_layers=2, latent_dims=[32, 16], allow_residual=False
        )

        model = MLP(config)
        x = torch.randn(4, 1)

        # Test training mode
        model.train()
        output_train = model(x)

        # Test evaluation mode
        model.eval()
        with torch.no_grad():
            output_eval = model(x)

        # Outputs should have same shape
        self.assertEqual(output_train.shape, output_eval.shape)
        self.assertTrue(torch.isfinite(output_train).all())
        self.assertTrue(torch.isfinite(output_eval).all())


if __name__ == "__main__":
    unittest.main()
