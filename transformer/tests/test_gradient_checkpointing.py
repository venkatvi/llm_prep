"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for gradient checkpointing FFN implementation.
"""

import unittest
import torch
import torch.nn as nn
import time
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gradient_checkpointing.ffn import (
    CheckpointedFFN, 
    CheckpointedFFNFunction, 
    AdaptiveCheckpointedFFN,
    memory_profiler,
    compare_ffn_implementations
)
from ffn import FFN


class TestCheckpointedFFNFunction(unittest.TestCase):
    """Test the custom autograd function for gradient checkpointing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embed_dim = 256
        self.latent_dim = 1024
        self.batch_size = 8
        self.seq_len = 128
        
        # Create layers
        self.layer_1 = nn.Linear(self.embed_dim, self.latent_dim)
        self.layer_2 = nn.Linear(self.latent_dim, self.embed_dim)
        self.activation = nn.ReLU()
        
        # Create test input
        self.input_tensor = torch.randn(
            self.batch_size, self.seq_len, self.embed_dim, requires_grad=True
        )
        
    def test_forward_pass(self):
        """Test that forward pass produces correct output."""
        # Use custom function
        output = CheckpointedFFNFunction.apply(
            self.input_tensor, self.layer_1, self.layer_2, self.activation
        )
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)
        
        # Check requires_grad is preserved
        self.assertTrue(output.requires_grad)
        
    def test_backward_pass(self):
        """Test that backward pass computes gradients correctly."""
        # Forward pass
        output = CheckpointedFFNFunction.apply(
            self.input_tensor, self.layer_1, self.layer_2, self.activation
        )
        
        # Backward pass
        target = torch.randn_like(output)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(self.input_tensor.grad)
        self.assertIsNotNone(self.layer_1.weight.grad)
        self.assertIsNotNone(self.layer_2.weight.grad)
        
    def test_gradient_correctness(self):
        """Test that gradients match standard PyTorch computation."""
        torch.manual_seed(42)
        
        # Standard computation
        layer_1_std = nn.Linear(self.embed_dim, self.latent_dim)
        layer_2_std = nn.Linear(self.latent_dim, self.embed_dim)
        activation_std = nn.ReLU()
        
        # Copy weights to ensure same computation
        layer_1_std.weight.data = self.layer_1.weight.data.clone()
        layer_1_std.bias.data = self.layer_1.bias.data.clone()
        layer_2_std.weight.data = self.layer_2.weight.data.clone()
        layer_2_std.bias.data = self.layer_2.bias.data.clone()
        
        input_std = self.input_tensor.clone().detach().requires_grad_(True)
        output_std = layer_2_std(activation_std(layer_1_std(input_std)))
        
        # Checkpointed computation
        input_checkpoint = self.input_tensor.clone().detach().requires_grad_(True)
        output_checkpoint = CheckpointedFFNFunction.apply(
            input_checkpoint, self.layer_1, self.layer_2, self.activation
        )
        
        # Compare outputs
        torch.testing.assert_close(output_std, output_checkpoint, rtol=1e-5, atol=1e-6)
        
        # Compare gradients
        target = torch.randn_like(output_std)
        
        loss_std = nn.MSELoss()(output_std, target)
        loss_std.backward()
        
        loss_checkpoint = nn.MSELoss()(output_checkpoint, target)
        loss_checkpoint.backward()
        
        torch.testing.assert_close(
            input_std.grad, input_checkpoint.grad, rtol=1e-4, atol=1e-5
        )


class TestCheckpointedFFN(unittest.TestCase):
    """Test the checkpointed FFN module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embed_dim = 256
        self.latent_dim = 1024
        self.batch_size = 8
        self.seq_len = 128
        
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
    def test_initialization(self):
        """Test FFN initialization."""
        ffn = CheckpointedFFN(self.embed_dim, self.latent_dim)
        
        self.assertEqual(ffn.layer_1.in_features, self.embed_dim)
        self.assertEqual(ffn.layer_1.out_features, self.latent_dim)
        self.assertEqual(ffn.layer_2.in_features, self.latent_dim)
        self.assertEqual(ffn.layer_2.out_features, self.embed_dim)
        self.assertTrue(ffn.enable_checkpointing)
        
    def test_activation_options(self):
        """Test different activation functions."""
        # ReLU
        ffn_relu = CheckpointedFFN(self.embed_dim, self.latent_dim, activation="relu")
        self.assertIsInstance(ffn_relu.activation, nn.ReLU)
        
        # GELU
        ffn_gelu = CheckpointedFFN(self.embed_dim, self.latent_dim, activation="gelu")
        self.assertIsInstance(ffn_gelu.activation, nn.GELU)
        
        # SwiGLU (SiLU)
        ffn_swish = CheckpointedFFN(self.embed_dim, self.latent_dim, activation="swish")
        self.assertIsInstance(ffn_swish.activation, nn.SiLU)
        
        # Invalid activation
        with self.assertRaises(ValueError):
            CheckpointedFFN(self.embed_dim, self.latent_dim, activation="invalid")
            
    def test_forward_training_mode(self):
        """Test forward pass in training mode."""
        ffn = CheckpointedFFN(self.embed_dim, self.latent_dim)
        ffn.train()
        
        output = ffn(self.input_tensor)
        expected_shape = (self.batch_size, self.seq_len, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)
        
    def test_forward_eval_mode(self):
        """Test forward pass in evaluation mode."""
        ffn = CheckpointedFFN(self.embed_dim, self.latent_dim)
        ffn.eval()
        
        output = ffn(self.input_tensor)
        expected_shape = (self.batch_size, self.seq_len, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)
        
    def test_checkpointing_disabled(self):
        """Test behavior when checkpointing is disabled."""
        ffn = CheckpointedFFN(
            self.embed_dim, self.latent_dim, enable_checkpointing=False
        )
        ffn.train()
        
        output = ffn(self.input_tensor)
        expected_shape = (self.batch_size, self.seq_len, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)
        
    def test_memory_stats(self):
        """Test memory statistics tracking."""
        ffn = CheckpointedFFN(self.embed_dim, self.latent_dim)
        
        # Initial stats should be zero
        stats = ffn.get_memory_stats()
        self.assertEqual(stats["forward_time"], 0)
        self.assertEqual(stats["recomputation_count"], 0)
        
        # Reset stats
        ffn.reset_stats()
        stats = ffn.get_memory_stats()
        self.assertEqual(stats["forward_time"], 0)
        
    def test_equivalence_with_standard_ffn(self):
        """Test that outputs are equivalent to standard FFN."""
        torch.manual_seed(42)
        
        # Standard FFN
        standard_ffn = FFN(self.embed_dim, self.latent_dim)
        
        # Checkpointed FFN with same weights
        checkpointed_ffn = CheckpointedFFN(self.embed_dim, self.latent_dim)
        checkpointed_ffn.layer_1.weight.data = standard_ffn.layer_1.weight.data.clone()
        checkpointed_ffn.layer_1.bias.data = standard_ffn.layer_1.bias.data.clone()
        checkpointed_ffn.layer_2.weight.data = standard_ffn.layer_2.weight.data.clone()
        checkpointed_ffn.layer_2.bias.data = standard_ffn.layer_2.bias.data.clone()
        
        # Disable checkpointing for exact comparison
        checkpointed_ffn.enable_checkpointing = False
        
        # Compare outputs
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        output_standard = standard_ffn(input_tensor)
        output_checkpointed = checkpointed_ffn(input_tensor)
        
        torch.testing.assert_close(
            output_standard, output_checkpointed, rtol=1e-5, atol=1e-6
        )


class TestAdaptiveCheckpointedFFN(unittest.TestCase):
    """Test adaptive checkpointing FFN."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embed_dim = 256
        self.latent_dim = 1024
        self.batch_size = 8
        self.seq_len = 128
        
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
    def test_initialization(self):
        """Test adaptive FFN initialization."""
        ffn = AdaptiveCheckpointedFFN(
            self.embed_dim, self.latent_dim, memory_threshold_mb=500.0
        )
        
        self.assertEqual(ffn.memory_threshold_mb, 500.0)
        self.assertFalse(ffn.enable_checkpointing)  # Initially disabled
        
    def test_forward_pass(self):
        """Test forward pass adapts checkpointing."""
        ffn = AdaptiveCheckpointedFFN(self.embed_dim, self.latent_dim)
        ffn.train()
        
        output = ffn(self.input_tensor)
        expected_shape = (self.batch_size, self.seq_len, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)


class TestMemoryProfiler(unittest.TestCase):
    """Test memory profiling utilities."""
    
    def test_memory_profiler_context(self):
        """Test memory profiler context manager."""
        # This test just ensures the context manager works
        # Actual memory measurement depends on CUDA availability
        with memory_profiler():
            # Some computation
            x = torch.randn(100, 100)
            y = torch.matmul(x, x.T)
            
        # Should complete without errors
        self.assertTrue(True)


class TestComparisonUtilities(unittest.TestCase):
    """Test comparison and benchmarking utilities."""
    
    def test_compare_ffn_implementations_cpu(self):
        """Test FFN comparison on CPU."""
        results = compare_ffn_implementations(
            embed_dim=128, latent_dim=512,
            batch_size=4, seq_len=64,
            num_iterations=2
        )
        
        # Should return results dict
        self.assertIsInstance(results, dict)
        self.assertIn("checkpointed_stats", results)
        
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_compare_ffn_implementations_cuda(self):
        """Test FFN comparison on CUDA."""
        results = compare_ffn_implementations(
            embed_dim=128, latent_dim=512,
            batch_size=4, seq_len=64,
            num_iterations=2
        )
        
        # Should return memory comparison results
        self.assertIsInstance(results, dict)
        self.assertIn("standard_peak_memory_mb", results)
        self.assertIn("checkpointed_peak_memory_mb", results)
        self.assertIn("memory_savings_percent", results)
        

class TestIntegration(unittest.TestCase):
    """Integration tests for gradient checkpointing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embed_dim = 256
        self.latent_dim = 1024
        self.batch_size = 8
        self.seq_len = 128
        
    def test_training_loop(self):
        """Test full training loop with gradient checkpointing."""
        # Create model and data
        model = CheckpointedFFN(self.embed_dim, self.latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        input_data = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        target_data = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        model.train()
        
        # Training step
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        self.assertTrue(True)
        
        # Check stats were updated
        stats = model.get_memory_stats()
        self.assertGreater(stats["forward_time"], 0)
        
    def test_gradient_accumulation(self):
        """Test gradient accumulation with checkpointed FFN."""
        model = CheckpointedFFN(self.embed_dim, self.latent_dim)
        criterion = nn.MSELoss()
        
        # Accumulate gradients over multiple batches
        accumulated_grad = None
        
        for i in range(3):
            input_data = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
            target_data = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
            
            output = model(input_data)
            loss = criterion(output, target_data) / 3  # Scale for accumulation
            loss.backward()
            
            if accumulated_grad is None:
                accumulated_grad = model.layer_1.weight.grad.clone()
            else:
                accumulated_grad += model.layer_1.weight.grad
                
        # Gradients should accumulate properly
        self.assertIsNotNone(model.layer_1.weight.grad)
        torch.testing.assert_close(
            model.layer_1.weight.grad, accumulated_grad, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    unittest.main(verbosity=2)