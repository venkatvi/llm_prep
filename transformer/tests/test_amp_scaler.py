"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for advanced gradient scaler implementation.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import math

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amp.scaler import (
    GradientScaler,
    ScalerConfig,
    ScalingStrategy,
    AMPTrainingLoop,
    create_optimized_scaler_config
)


class TestScalerConfig(unittest.TestCase):
    """Test ScalerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ScalerConfig()
        
        self.assertEqual(config.init_scale, 2**16)
        self.assertEqual(config.growth_factor, 2.0)
        self.assertEqual(config.backoff_factor, 0.5)
        self.assertEqual(config.growth_interval, 2000)
        self.assertEqual(config.strategy, ScalingStrategy.DYNAMIC)
        self.assertFalse(config.verbose)
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ScalerConfig(
            init_scale=1000.0,
            growth_factor=1.5,
            strategy=ScalingStrategy.ADAPTIVE,
            verbose=True
        )
        
        self.assertEqual(config.init_scale, 1000.0)
        self.assertEqual(config.growth_factor, 1.5)
        self.assertEqual(config.strategy, ScalingStrategy.ADAPTIVE)
        self.assertTrue(config.verbose)


class TestGradientScaler(unittest.TestCase):
    """Test GradientScaler implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ScalerConfig(
            init_scale=1024.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=10,  # Small for testing
            verbose=False
        )
        self.scaler = GradientScaler(self.config)
        
        # Create simple model and optimizer
        self.model = nn.Linear(10, 5)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
    def test_initialization(self):
        """Test scaler initialization."""
        self.assertEqual(self.scaler.get_scale(), 1024.0)
        self.assertEqual(self.scaler._growth_tracker, 0)
        self.assertEqual(self.scaler._current_step, 0)
        
        stats = self.scaler.get_stats()
        self.assertEqual(stats["total_steps"], 0)
        self.assertEqual(stats["overflow_steps"], 0)
        
    def test_scale_tensor(self):
        """Test tensor scaling."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        scaled = self.scaler.scale(tensor)
        expected = tensor * 1024.0
        
        torch.testing.assert_close(scaled, expected)
        
    def test_unscale_gradients(self):
        """Test gradient unscaling."""
        # Create gradients
        x = torch.randn(5, 10, requires_grad=True)
        y = torch.randn(5, 5)
        
        output = self.model(x)
        loss = nn.MSELoss()(output, y)
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        
        # Check gradients are scaled
        original_grad = self.model.weight.grad.clone()
        
        # Unscale gradients
        self.scaler.unscale_(self.optimizer)
        
        # Check gradients are unscaled
        expected_grad = original_grad / 1024.0
        torch.testing.assert_close(self.model.weight.grad, expected_grad, rtol=1e-5, atol=1e-6)
        
    def test_successful_step(self):
        """Test successful optimizer step."""
        x = torch.randn(5, 10)
        y = torch.randn(5, 5)
        
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = nn.MSELoss()(output, y)
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        
        # Take step
        success = self.scaler.step(self.optimizer)
        self.assertTrue(success)
        
        stats = self.scaler.get_stats()
        self.assertEqual(stats["total_steps"], 1)
        self.assertEqual(stats["successful_steps"], 1)
        self.assertEqual(stats["overflow_steps"], 0)
        
    def test_overflow_detection(self):
        """Test inf/NaN gradient detection and handling."""
        x = torch.randn(5, 10)
        y = torch.randn(5, 5)
        
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = nn.MSELoss()(output, y)
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        
        # Inject inf gradient
        self.model.weight.grad[0, 0] = float('inf')
        
        # Take step should fail
        success = self.scaler.step(self.optimizer)
        self.assertFalse(success)
        
        stats = self.scaler.get_stats()
        self.assertEqual(stats["overflow_steps"], 1)
        self.assertEqual(stats["successful_steps"], 0)
        
    def test_scale_reduction_on_overflow(self):
        """Test scale reduction when overflow occurs."""
        initial_scale = self.scaler.get_scale()
        
        # Simulate overflow
        self.scaler._handle_overflow()
        
        new_scale = self.scaler.get_scale()
        expected_scale = initial_scale * self.config.backoff_factor
        
        self.assertAlmostEqual(new_scale, expected_scale, places=5)
        
    def test_scale_growth(self):
        """Test scale growth after stable iterations."""
        initial_scale = self.scaler.get_scale()
        
        # Simulate stable steps
        for _ in range(self.config.growth_interval + 1):
            self.scaler._stable_iterations += 1
            self.scaler._growth_tracker += 1
            self.scaler.update()
            
        new_scale = self.scaler.get_scale()
        self.assertGreater(new_scale, initial_scale)
        
    def test_recovery_mode(self):
        """Test recovery mode after consecutive overflows."""
        # Simulate multiple consecutive overflows
        for _ in range(3):
            self.scaler._handle_overflow()
            
        stats = self.scaler.get_stats()
        self.assertGreater(stats["recovery_events"], 0)
        self.assertTrue(self.scaler._in_recovery_mode)
        
    def test_adaptive_scaling_strategy(self):
        """Test adaptive scaling strategy."""
        config = ScalerConfig(strategy=ScalingStrategy.ADAPTIVE)
        scaler = GradientScaler(config)
        
        # Add some overflow history
        scaler._overflow_history = [True] * 5 + [False] * 15  # 25% overflow rate
        
        initial_scale = scaler.get_scale()
        scaler._update_scale_adaptive()
        
        # Scale should be reduced due to high overflow rate
        self.assertLess(scaler.get_scale(), initial_scale)
        
    def test_conservative_scaling_strategy(self):
        """Test conservative scaling strategy."""
        config = ScalerConfig(
            strategy=ScalingStrategy.CONSERVATIVE,
            stability_threshold=5
        )
        scaler = GradientScaler(config)
        
        initial_scale = scaler.get_scale()
        
        # Simulate very stable training
        scaler._stable_iterations = 15  # More than 2x threshold
        scaler._consecutive_overflows = 0
        scaler._total_overflows = 0
        
        scaler._update_scale_conservative()
        
        # Scale should increase very slightly
        self.assertGreaterEqual(scaler.get_scale(), initial_scale)
        
    def test_dynamic_frequency_adjustment(self):
        """Test dynamic update frequency adjustment."""
        config = ScalerConfig(enable_dynamic_frequency=True)
        scaler = GradientScaler(config)
        
        # High overflow rate should increase growth interval
        scaler._overflow_history = [True] * 5 + [False] * 15
        initial_interval = scaler._current_growth_interval
        
        scaler._adjust_update_frequency()
        
        self.assertGreater(scaler._current_growth_interval, initial_interval)
        
    def test_state_dict_operations(self):
        """Test state dict save and load."""
        # Modify scaler state
        self.scaler._scale = torch.tensor(2048.0)
        self.scaler._growth_tracker = 5
        self.scaler._current_step = 10
        self.scaler.stats["total_steps"] = 15
        
        # Save state
        state_dict = self.scaler.state_dict()
        
        # Create new scaler and load state
        new_scaler = GradientScaler(self.config)
        new_scaler.load_state_dict(state_dict)
        
        # Check state is preserved
        self.assertEqual(new_scaler.get_scale(), 2048.0)
        self.assertEqual(new_scaler._growth_tracker, 5)
        self.assertEqual(new_scaler._current_step, 10)
        self.assertEqual(new_scaler.stats["total_steps"], 15)
        
    def test_comprehensive_stats(self):
        """Test comprehensive statistics collection."""
        # Simulate various training scenarios
        self.scaler._stable_iterations = 100
        self.scaler._unstable_iterations = 10
        self.scaler._total_overflows = 5
        self.scaler.stats["total_steps"] = 200
        self.scaler.stats["successful_steps"] = 190
        self.scaler.stats["overflow_steps"] = 10
        
        stats = self.scaler.get_stats()
        
        # Check all expected fields are present
        expected_fields = [
            "current_scale", "current_strategy", "stable_iterations",
            "unstable_iterations", "success_rate", "overflow_rate"
        ]
        
        for field in expected_fields:
            self.assertIn(field, stats)
            
        # Check calculated metrics
        self.assertAlmostEqual(stats["success_rate"], 190/200, places=5)
        self.assertAlmostEqual(stats["overflow_rate"], 10/200, places=5)


class TestAMPTrainingLoop(unittest.TestCase):
    """Test AMPTrainingLoop wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Linear(10, 5)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        
        config = ScalerConfig(init_scale=1024.0, verbose=False)
        self.training_loop = AMPTrainingLoop(
            self.model, self.optimizer, config, enable_autocast=False
        )
        
    def test_initialization(self):
        """Test training loop initialization."""
        self.assertEqual(self.training_loop.model, self.model)
        self.assertEqual(self.training_loop.optimizer, self.optimizer)
        self.assertIsInstance(self.training_loop.scaler, GradientScaler)
        
    def test_train_step_success(self):
        """Test successful training step."""
        # Create batch data
        inputs = torch.randn(8, 10)
        targets = torch.randn(8, 5)
        batch_data = (inputs, targets)
        
        # Perform training step
        result = self.training_loop.train_step(batch_data, self.criterion)
        
        # Check results
        self.assertTrue(result["step_successful"])
        self.assertIsInstance(result["loss"], float)
        self.assertFalse(math.isinf(result["loss"]))
        self.assertGreater(result["current_scale"], 0)
        
    def test_train_step_with_custom_forward(self):
        """Test training step with custom forward function."""
        inputs = torch.randn(8, 10)
        
        def custom_forward(data):
            outputs = self.model(data)
            loss = outputs.sum()  # Simple loss
            return loss, outputs
            
        result = self.training_loop.train_step(
            inputs, self.criterion, forward_fn=custom_forward
        )
        
        self.assertTrue(result["step_successful"])
        self.assertIsNotNone(result["outputs"])
        
    def test_comprehensive_stats(self):
        """Test comprehensive statistics collection."""
        # Run several training steps
        for _ in range(5):
            inputs = torch.randn(4, 10)
            targets = torch.randn(4, 5)
            batch_data = (inputs, targets)
            
            self.training_loop.train_step(batch_data, self.criterion)
            
        stats = self.training_loop.get_comprehensive_stats()
        
        # Check structure
        self.assertIn("training", stats)
        self.assertIn("scaler", stats)
        
        # Check training stats
        training_stats = stats["training"]
        self.assertEqual(training_stats["batches_processed"], 5)
        self.assertIn("avg_loss", training_stats)
        self.assertIn("overflow_rate", training_stats)


class TestScalerConfigCreation(unittest.TestCase):
    """Test optimized scaler configuration creation."""
    
    def test_low_instability_config(self):
        """Test configuration for low instability training."""
        config = create_optimized_scaler_config("low")
        
        self.assertEqual(config.strategy, ScalingStrategy.DYNAMIC)
        self.assertEqual(config.init_scale, 2**16)
        self.assertEqual(config.growth_interval, 1000)
        
    def test_medium_instability_config(self):
        """Test configuration for medium instability training."""
        config = create_optimized_scaler_config("medium")
        
        self.assertEqual(config.strategy, ScalingStrategy.ADAPTIVE)
        self.assertEqual(config.init_scale, 2**14)
        self.assertTrue(config.enable_dynamic_frequency)
        
    def test_high_instability_config(self):
        """Test configuration for high instability training."""
        config = create_optimized_scaler_config("high")
        
        self.assertEqual(config.strategy, ScalingStrategy.CONSERVATIVE)
        self.assertEqual(config.init_scale, 2**12)
        self.assertEqual(config.recovery_factor, 0.05)
        self.assertGreater(config.stability_threshold, 1000)


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic training scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        
    def test_stable_training_scenario(self):
        """Test stable training with gradual scale increases."""
        config = create_optimized_scaler_config("low")
        scaler = GradientScaler(config)
        
        initial_scale = scaler.get_scale()
        successful_steps = 0
        
        # Simulate stable training
        for step in range(50):
            # Generate batch
            inputs = torch.randn(16, 20)
            targets = torch.randint(0, 10, (16,))
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            
            if scaler.step(self.optimizer):
                successful_steps += 1
                
            scaler.update()
            
        # Should have mostly successful steps and increased scale
        self.assertGreater(successful_steps, 45)
        
        final_stats = scaler.get_stats()
        self.assertGreater(final_stats["success_rate"], 0.9)
        
    def test_unstable_training_scenario(self):
        """Test training with overflow recovery."""
        config = create_optimized_scaler_config("high")
        scaler = GradientScaler(config)
        
        recovery_events = 0
        
        # Simulate unstable training
        for step in range(30):
            inputs = torch.randn(16, 20)
            targets = torch.randint(0, 10, (16,))
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            
            # Occasionally inject instability after gradients are created
            if step % 8 == 0:
                # Force overflow by making gradients inf
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.fill_(float('inf'))
            
            scaler.step(self.optimizer)
            scaler.update()
            
            if scaler._in_recovery_mode:
                recovery_events += 1
                
        # Should have handled overflows gracefully
        final_stats = scaler.get_stats()
        self.assertGreater(final_stats["overflow_steps"], 0)
        self.assertGreater(final_stats["scale_reductions"], 0)
        
    def test_mixed_precision_compatibility(self):
        """Test compatibility with PyTorch's autocast."""
        config = ScalerConfig(verbose=False)
        training_loop = AMPTrainingLoop(
            self.model, self.optimizer, config, enable_autocast=True
        )
        
        # Test with autocast enabled
        inputs = torch.randn(8, 20)
        targets = torch.randint(0, 10, (8,))
        batch_data = (inputs, targets)
        
        result = training_loop.train_step(batch_data, self.criterion)
        
        # Should work without errors
        self.assertIsInstance(result["loss"], float)
        self.assertTrue(result["step_successful"] or math.isinf(result["loss"]))


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    unittest.main(verbosity=2)