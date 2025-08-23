"""
Integration tests for regression experiments.
"""

import os
import sys
import tempfile
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.configs import ExperimentConfig, TrainConfig
from lib.train import TrainContext
from regression.configs import RegressionModelConfig
from regression.dataset import generate_polynomial_data, prepare_data
from regression.e_linear_reg import LinearRegressionModel
from regression.e_non_linear_reg import MLP
from regression.experiment import RegressionExperiment


class TestRegressionIntegration(unittest.TestCase):
    """Integration tests for complete regression workflows."""

    def setUp(self):
        """Set up common test configurations."""
        self.train_config = TrainConfig(
            epochs=2,  # Very small for testing
            optimizer="adam",
            lr=0.01,
            lr_scheduler="none",
            custom_loss="mse"
        )

    def test_linear_regression_small_experiment(self):
        """Test complete linear regression experiment workflow."""
        # Generate small dataset
        inputs, targets = generate_polynomial_data(
            num_samples=20, degree=1, noise_level=0.1, random_seed=42
        )
        
        # Create model config
        model_config = RegressionModelConfig(
            name="linear_test",
            custom_act="relu",
            num_latent_layers=1,
            latent_dims=[16],
            allow_residual=False
        )
        
        # Create experiment config
        experiment_config = ExperimentConfig(
            name="test_linear_exp",
            type="linear",
            model=model_config,
            train_config=self.train_config
        )
        
        # Create and run experiment
        experiment = RegressionExperiment(experiment_config)
        experiment.generate_data(inputs, targets)
        
        # Test training
        initial_loss = float('inf')
        try:
            experiment.train()
            self.assertIsNotNone(experiment.train_loss)
            self.assertIsNotNone(experiment.val_loss)
            self.assertTrue(experiment.train_loss > 0)
            self.assertTrue(experiment.val_loss > 0)
        except Exception as e:
            self.fail(f"Training failed: {e}")
        
        # Test prediction
        try:
            predictions = experiment.predict()
            self.assertIsInstance(predictions, torch.Tensor)
            self.assertEqual(predictions.shape[1], 1)  # Output dimension
            self.assertTrue(torch.isfinite(predictions).all())
        except Exception as e:
            self.fail(f"Prediction failed: {e}")

    def test_nonlinear_regression_small_experiment(self):
        """Test complete non-linear regression experiment workflow."""
        # Generate non-linear dataset
        inputs, targets = generate_polynomial_data(
            num_samples=30, degree=2, noise_level=0.05, random_seed=123
        )
        
        # Create model config
        model_config = RegressionModelConfig(
            name="nonlinear_test",
            custom_act="tanh",
            num_latent_layers=2,
            latent_dims=[32, 16],
            allow_residual=False
        )
        
        # Create experiment config
        experiment_config = ExperimentConfig(
            name="test_nonlinear_exp",
            type="nlinear",
            model=model_config,
            train_config=self.train_config
        )
        
        # Create and run experiment
        experiment = RegressionExperiment(experiment_config)
        experiment.generate_data(inputs, targets)
        
        # Test training
        try:
            experiment.train()
            self.assertIsNotNone(experiment.train_loss)
            self.assertIsNotNone(experiment.val_loss)
        except Exception as e:
            self.fail(f"Non-linear training failed: {e}")
        
        # Test prediction
        try:
            predictions = experiment.predict()
            self.assertIsInstance(predictions, torch.Tensor)
            self.assertTrue(torch.isfinite(predictions).all())
        except Exception as e:
            self.fail(f"Non-linear prediction failed: {e}")

    def test_model_saving_and_loading(self):
        """Test model checkpoint saving and loading."""
        # Generate small dataset
        inputs, targets = generate_polynomial_data(
            num_samples=15, degree=1, random_seed=42
        )
        
        model_config = RegressionModelConfig(
            name="save_test",
            custom_act="relu",
            num_latent_layers=1,
            latent_dims=[8],
            allow_residual=False
        )
        
        experiment_config = ExperimentConfig(
            name="test_save_exp",
            type="linear",
            model=model_config,
            train_config=self.train_config
        )
        
        experiment = RegressionExperiment(experiment_config)
        experiment.generate_data(inputs, targets)
        
        # Train and save
        experiment.train()
        original_loss = experiment.train_loss
        
        # Get model state before saving
        original_params = {}
        for name, param in experiment.model.named_parameters():
            original_params[name] = param.clone()
        
        try:
            experiment.save()
            
            # Verify checkpoint file exists
            checkpoint_path = f"{experiment_config.name}.ckpt"
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # Create new experiment and load
            new_experiment = RegressionExperiment(experiment_config)
            new_experiment.generate_data(inputs, targets)
            new_experiment.load(checkpoint_path)
            
            # Verify loaded state matches original
            for name, param in new_experiment.model.named_parameters():
                torch.testing.assert_close(param, original_params[name])
            
            # Clean up
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
                
        except Exception as e:
            self.fail(f"Save/load failed: {e}")

    def test_data_loader_integration(self):
        """Test integration with data loaders."""
        # Generate data
        inputs, targets = generate_polynomial_data(
            num_samples=24, degree=1, random_seed=42  # Multiple of batch size
        )
        
        # Create data loader
        dataloader, csv_path = prepare_data(inputs, targets, '.csv', batch_size=8)
        
        try:
            # Test data loader works
            batch_count = 0
            total_samples = 0
            
            for batch_inputs, batch_targets in dataloader:
                batch_count += 1
                total_samples += len(batch_inputs)
                
                # Verify batch shapes
                self.assertEqual(batch_inputs.shape[1], 1)
                self.assertEqual(batch_targets.shape[1], 1)
                self.assertLessEqual(len(batch_inputs), 8)  # batch size
            
            self.assertEqual(total_samples, 24)  # All samples processed
            self.assertGreater(batch_count, 1)  # Multiple batches
            
        finally:
            # Clean up
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_different_optimizers_and_schedulers(self):
        """Test experiments with different optimizers and schedulers."""
        inputs, targets = generate_polynomial_data(
            num_samples=20, degree=1, random_seed=42
        )
        
        model_config = RegressionModelConfig(
            name="optimizer_test",
            custom_act="relu",
            num_latent_layers=1,
            latent_dims=[16],
            allow_residual=False
        )
        
        # Test different optimizer/scheduler combinations
        test_configs = [
            {"optimizer": "adam", "lr_scheduler": "none"},
            {"optimizer": "sgd", "lr_scheduler": "none"},
            {"optimizer": "adam", "lr_scheduler": "step"}
        ]
        
        for config in test_configs:
            with self.subTest(**config):
                train_config = TrainConfig(
                    epochs=2,
                    optimizer=config["optimizer"],
                    lr=0.01,
                    lr_scheduler=config["lr_scheduler"],
                    custom_loss="mse"
                )
                
                experiment_config = ExperimentConfig(
                    name=f"test_{config['optimizer']}_{config['lr_scheduler']}",
                    type="linear",
                    model=model_config,
                    train_config=train_config
                )
                
                experiment = RegressionExperiment(experiment_config)
                experiment.generate_data(inputs, targets)
                
                try:
                    experiment.train()
                    self.assertIsNotNone(experiment.train_loss)
                    self.assertTrue(experiment.train_loss > 0)
                except Exception as e:
                    self.fail(f"Training failed with {config}: {e}")

    def test_loss_functions(self):
        """Test experiments with different loss functions."""
        inputs, targets = generate_polynomial_data(
            num_samples=20, degree=1, random_seed=42
        )
        
        model_config = RegressionModelConfig(
            name="loss_test",
            custom_act="relu",
            num_latent_layers=1,
            latent_dims=[16],
            allow_residual=False
        )
        
        loss_functions = ["mse", "mae", "huber"]
        
        for loss_fn in loss_functions:
            with self.subTest(loss_function=loss_fn):
                train_config = TrainConfig(
                    epochs=2,
                    optimizer="adam",
                    lr=0.01,
                    lr_scheduler="none",
                    custom_loss=loss_fn
                )
                
                experiment_config = ExperimentConfig(
                    name=f"test_loss_{loss_fn}",
                    type="linear",
                    model=model_config,
                    train_config=train_config
                )
                
                experiment = RegressionExperiment(experiment_config)
                experiment.generate_data(inputs, targets)
                
                try:
                    experiment.train()
                    predictions = experiment.predict()
                    
                    self.assertIsNotNone(experiment.train_loss)
                    self.assertIsInstance(predictions, torch.Tensor)
                    self.assertTrue(torch.isfinite(predictions).all())
                    
                except Exception as e:
                    self.fail(f"Training with {loss_fn} loss failed: {e}")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small dataset
        inputs, targets = generate_polynomial_data(
            num_samples=3, degree=1, random_seed=42
        )
        
        model_config = RegressionModelConfig(
            name="edge_case_test",
            custom_act="relu",
            num_latent_layers=1,
            latent_dims=[8],
            allow_residual=False
        )
        
        experiment_config = ExperimentConfig(
            name="test_edge_case",
            type="linear",
            model=model_config,
            train_config=self.train_config
        )
        
        experiment = RegressionExperiment(experiment_config)
        experiment.generate_data(inputs, targets)
        
        # Should handle small dataset gracefully
        try:
            experiment.train()
            predictions = experiment.predict()
            
            self.assertIsInstance(predictions, torch.Tensor)
            self.assertTrue(torch.isfinite(predictions).all())
            
        except Exception as e:
            self.fail(f"Edge case handling failed: {e}")

    def test_experiment_reproducibility(self):
        """Test that experiments are reproducible with same seed."""
        inputs, targets = generate_polynomial_data(
            num_samples=20, degree=1, random_seed=42
        )
        
        model_config = RegressionModelConfig(
            name="repro_test",
            custom_act="relu",
            num_latent_layers=1,
            latent_dims=[16],
            allow_residual=False
        )
        
        experiment_config = ExperimentConfig(
            name="test_reproducibility",
            type="linear",
            model=model_config,
            train_config=self.train_config
        )
        
        # Run experiment twice with same config
        results1 = self._run_experiment_and_get_results(experiment_config, inputs, targets)
        results2 = self._run_experiment_and_get_results(experiment_config, inputs, targets)
        
        # Results should be similar (allowing for some variance due to initialization)
        self.assertAlmostEqual(results1["train_loss"], results2["train_loss"], places=1)
        self.assertEqual(results1["predictions"].shape, results2["predictions"].shape)

    def _run_experiment_and_get_results(self, config, inputs, targets):
        """Helper method to run experiment and return results."""
        experiment = RegressionExperiment(config)
        experiment.generate_data(inputs, targets)
        experiment.train()
        predictions = experiment.predict()
        
        return {
            "train_loss": experiment.train_loss,
            "val_loss": experiment.val_loss,
            "predictions": predictions
        }


if __name__ == "__main__":
    unittest.main()