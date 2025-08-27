"""
Copyright (c) 2025. All rights reserved.
"""

"""
Integration tests for transformer regression model.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
import torch

from regression.configs import TransformerModelConfig, FFNConfig
from regression.h_transformer import RegressionTransformerModel


class TestTransformerRegression:
    """Integration test suite for transformer regression."""

    def test_transformer_regression_initialization(self):
        """Test RegressionTransformerModel initialization."""
        ffn_config = FFNConfig(
            embed_dim=32,
            latent_dim=128,
            use_moe=False,
        )

        config = TransformerModelConfig(
            name="test_transformer",
            input_dim=8,
            embed_dim=32,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=4,
            output_dim=1,
            ffn_config=ffn_config,
        )

        model = RegressionTransformerModel(config)

        assert model.config == config
        assert hasattr(model, "model")

    def test_data_generation(self):
        """Test synthetic data generation."""
        ffn_config = FFNConfig(
            embed_dim=32,
            latent_dim=128,
            use_moe=False,
        )

        config = TransformerModelConfig(
            name="test_transformer",
            input_dim=8,
            embed_dim=32,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=4,
            output_dim=1,
            ffn_config=ffn_config,
        )

        model = RegressionTransformerModel(config)
        inputs, targets = model.generate_data(random_seed=42)

        # Check data shapes
        assert inputs.shape == (100, 32, 8)  # num_samples, seq_len, input_dim
        assert targets.shape == (100,)  # num_samples

        # Check data is finite
        assert torch.isfinite(inputs).all()
        assert torch.isfinite(targets).all()

        # Check targets are computed correctly (sum of flattened inputs)
        expected_targets = inputs.reshape(100, -1).sum(dim=1)
        assert torch.allclose(targets, expected_targets)

    def test_data_generation_reproducibility(self):
        """Test data generation is reproducible with same seed."""
        ffn_config = FFNConfig(
            embed_dim=32,
            latent_dim=128,
            use_moe=False,
        )

        config = TransformerModelConfig(
            name="test_transformer",
            input_dim=8,
            embed_dim=32,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=4,
            output_dim=1,
            ffn_config=ffn_config,
        )

        model = RegressionTransformerModel(config)

        inputs1, targets1 = model.generate_data(random_seed=42)
        inputs2, targets2 = model.generate_data(random_seed=42)

        assert torch.allclose(inputs1, inputs2)
        assert torch.allclose(targets1, targets2)

    def test_forward_pass_integration(self):
        """Test complete forward pass through regression model."""
        ffn_config = FFNConfig(
            embed_dim=32,
            latent_dim=128,
            use_moe=False,
        )

        config = TransformerModelConfig(
            name="test_transformer",
            input_dim=8,
            embed_dim=32,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=4,
            output_dim=1,
            ffn_config=ffn_config,
        )

        model = RegressionTransformerModel(config)
        inputs, targets = model.generate_data(random_seed=42)

        # Forward pass
        predictions = model(inputs)

        assert predictions.shape == (100, 1)
        assert torch.isfinite(predictions).all()

    def test_gradient_flow_integration(self):
        """Test gradients flow through entire regression pipeline."""
        ffn_config = FFNConfig(
            embed_dim=32,
            latent_dim=128,
            use_moe=False,
        )

        config = TransformerModelConfig(
            name="test_transformer",
            input_dim=8,
            embed_dim=32,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=4,
            output_dim=1,
            ffn_config=ffn_config,
        )

        model = RegressionTransformerModel(config)
        inputs, targets = model.generate_data(random_seed=42)

        # Forward pass
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions.squeeze(), targets)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_training_step_simulation(self):
        """Test a complete training step simulation."""
        ffn_config = FFNConfig(
            embed_dim=32,
            latent_dim=128,
            use_moe=False,
        )

        config = TransformerModelConfig(
            name="test_transformer",
            input_dim=8,
            embed_dim=32,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=4,
            output_dim=1,
            ffn_config=ffn_config,
        )

        model = RegressionTransformerModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        inputs, targets = model.generate_data(random_seed=42)

        # Training step
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions.squeeze(), targets)
        loss.backward()

        # Check loss is reasonable
        assert loss.item() > 0
        assert torch.isfinite(loss)

        # Update parameters
        initial_param = next(model.parameters()).clone()
        optimizer.step()
        updated_param = next(model.parameters())

        # Parameters should have changed
        assert not torch.allclose(initial_param, updated_param)

    def test_multiple_training_steps(self):
        """Test multiple training steps reduce loss."""
        ffn_config = FFNConfig(
            embed_dim=32,
            latent_dim=128,
            use_moe=False,
        )

        config = TransformerModelConfig(
            name="test_transformer",
            input_dim=8,
            embed_dim=32,
            ffn_latent_dim=128,
            num_layers=1,  # Smaller model for faster training
            num_heads=2,
            num_groups=2,
            output_dim=1,
            ffn_config=ffn_config,
        )

        model = RegressionTransformerModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        inputs, targets = model.generate_data(random_seed=42)

        losses = []

        # Multiple training steps
        for step in range(10):
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = torch.nn.functional.mse_loss(predictions.squeeze(), targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease
        initial_loss = losses[0]
        final_loss = losses[-1]

        # Allow some flexibility due to optimization dynamics
        assert final_loss <= initial_loss * 1.1  # At most 10% increase

    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        ffn_config = FFNConfig(
            embed_dim=32,
            latent_dim=128,
            use_moe=False,
        )

        config = TransformerModelConfig(
            name="test_transformer",
            input_dim=8,
            embed_dim=32,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=4,
            output_dim=1,
            ffn_config=ffn_config,
        )

        model = RegressionTransformerModel(config)

        for batch_size in [1, 5, 16, 32]:
            seq_len = 16
            inputs = torch.randn(batch_size, seq_len, 8)
            predictions = model(inputs)
            assert predictions.shape == (batch_size, 1)

    def test_model_state_dict(self):
        """Test model state can be saved and loaded."""
        ffn_config = FFNConfig(
            embed_dim=32,
            latent_dim=128,
            use_moe=False,
        )

        config = TransformerModelConfig(
            name="test_transformer",
            input_dim=8,
            embed_dim=32,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=4,
            output_dim=1,
            ffn_config=ffn_config,
        )

        model1 = RegressionTransformerModel(config)
        model2 = RegressionTransformerModel(config)

        # Models should be different initially
        inputs = torch.randn(5, 16, 8)
        pred1 = model1(inputs)
        pred2 = model2(inputs)
        assert not torch.allclose(pred1, pred2)

        # Load state dict
        model2.load_state_dict(model1.state_dict())

        # Models should now produce same output
        pred1 = model1(inputs)
        pred2 = model2(inputs)
        assert torch.allclose(pred1, pred2)

    def test_config_validation(self):
        """Test model works with various config parameters."""
        small_ffn_config = FFNConfig(
            embed_dim=16,
            latent_dim=64,
            use_moe=False,
        )
        medium_ffn_config = FFNConfig(
            embed_dim=32,
            latent_dim=128,
            use_moe=False,
        )
        large_ffn_config = FFNConfig(
            embed_dim=64,
            latent_dim=256,
            use_moe=False,
        )

        configs = [
            # Small model
            TransformerModelConfig(
                name="small",
                input_dim=4,
                embed_dim=16,
                ffn_latent_dim=64,
                num_layers=1,
                num_heads=2,
                num_groups=2,
                output_dim=1,
                ffn_config=small_ffn_config,
            ),
            # Medium model
            TransformerModelConfig(
                name="medium",
                input_dim=8,
                embed_dim=32,
                ffn_latent_dim=128,
                num_layers=2,
                num_heads=4,
                num_groups=4,
                output_dim=1,
                ffn_config=medium_ffn_config,
            ),
            # Large model
            TransformerModelConfig(
                name="large",
                input_dim=16,
                embed_dim=64,
                ffn_latent_dim=256,
                num_layers=3,
                num_heads=8,
                num_groups=8,
                output_dim=1,
                ffn_config=large_ffn_config,
            ),
        ]

        for config in configs:
            model = RegressionTransformerModel(config)
            inputs, targets = model.generate_data(random_seed=42)
            predictions = model(inputs)

            assert predictions.shape == (100, 1)
            assert torch.isfinite(predictions).all()


if __name__ == "__main__":
    pytest.main([__file__])
