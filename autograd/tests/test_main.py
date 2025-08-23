"""
Copyright (c) 2025. All rights reserved.
"""

"""
Integration tests for the complete autograd pipeline.

This module tests the integration of multiple custom autograd functions
as demonstrated in main.py, including the combination of linear layers
and learnable activation functions.
"""

import os
import sys
import unittest

import torch

# Add parent directory to path to import autograd modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activations import LearnedSiLU
from linear import Linear


class TestIntegration(unittest.TestCase):
    """Test integration of multiple custom autograd functions"""

    def test_linear_plus_activation(self):
        """Test Linear layer followed by LearnedSiLU activation"""
        # Setup inputs similar to main.py
        x = torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True)
        w = torch.tensor([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]], requires_grad=True)
        b = torch.tensor([[-5.0, -12.0]], requires_grad=True)
        slope = torch.tensor([-1.0], requires_grad=True)

        # Forward pass
        y = Linear.apply(x, w, b)
        z = LearnedSiLU.apply(y, slope)

        # Check intermediate result
        # y = [2,2,2] @ [[3,1],[3,1],[3,1]] + [-5,-12] = [18,6] + [-5,-12] = [13,-6]
        expected_y = torch.tensor([[13.0, -6.0]])
        torch.testing.assert_close(y, expected_y, rtol=1e-4, atol=1e-4)

        # z should be computed but we'll mainly test that gradients flow correctly
        self.assertEqual(z.shape, (1, 2))

        # Backward pass
        loss = z.sum()
        loss.backward()

        # Check that all gradients are computed
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertIsNotNone(b.grad)
        self.assertIsNotNone(slope.grad)

        # Check gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(w.grad.shape, w.shape)
        self.assertEqual(b.grad.shape, b.shape)
        self.assertEqual(slope.grad.shape, slope.shape)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the entire network"""
        batch_size, input_dim, output_dim = 2, 3, 4

        x = torch.randn(batch_size, input_dim, requires_grad=True)
        w = torch.randn(output_dim, input_dim, requires_grad=True)
        b = torch.randn(batch_size, output_dim, requires_grad=True)
        slope = torch.randn(1, requires_grad=True)

        # Forward pass through network
        linear_out = Linear.apply(x, w, b)
        final_out = LearnedSiLU.apply(linear_out, slope)

        # Scalar loss for backward
        loss = final_out.sum()

        # Clear any existing gradients
        for param in [x, w, b, slope]:
            if param.grad is not None:
                param.grad.zero_()

        # Backward pass
        loss.backward()

        # Verify all gradients are non-None and have correct shapes
        for param in [x, w, b, slope]:
            self.assertIsNotNone(
                param.grad, f"Gradient is None for parameter with shape {param.shape}"
            )
            self.assertEqual(param.grad.shape, param.shape)

        # Verify gradients are not all zeros (for random inputs, this should be true)
        for param in [x, w, b, slope]:
            self.assertFalse(
                torch.allclose(param.grad, torch.zeros_like(param.grad)),
                f"Gradient is all zeros for parameter with shape {param.shape}",
            )

    def test_multiple_forward_backward(self):
        """Test multiple forward/backward passes"""
        x = torch.randn(1, 2, requires_grad=True)
        w = torch.randn(3, 2, requires_grad=True)
        b = torch.randn(1, 3, requires_grad=True)
        slope = torch.randn(1, requires_grad=True)

        for i in range(3):
            # Clear gradients
            for param in [x, w, b, slope]:
                if param.grad is not None:
                    param.grad.zero_()

            # Forward pass
            y = Linear.apply(x, w, b)
            z = LearnedSiLU.apply(y, slope)
            loss = z.sum()

            # Backward pass
            loss.backward()

            # Check gradients exist
            for param in [x, w, b, slope]:
                self.assertIsNotNone(param.grad)

    def test_gradient_accumulation(self):
        """Test gradient accumulation across multiple backward passes"""
        x = torch.randn(1, 2, requires_grad=True)
        w = torch.randn(2, 2, requires_grad=True)
        b = torch.randn(1, 2, requires_grad=True)
        slope = torch.randn(1, requires_grad=True)

        # First backward pass
        y1 = Linear.apply(x, w, b)
        z1 = LearnedSiLU.apply(y1, slope)
        loss1 = z1.sum()
        loss1.backward()

        # Store first gradients
        first_grads = {
            "x": x.grad.clone(),
            "w": w.grad.clone(),
            "b": b.grad.clone(),
            "slope": slope.grad.clone(),
        }

        # Second backward pass (without zeroing gradients)
        y2 = Linear.apply(x, w, b)
        z2 = LearnedSiLU.apply(y2, slope)
        loss2 = z2.sum()
        loss2.backward()

        # Gradients should be accumulated (doubled)
        for param_name, param in [("x", x), ("w", w), ("b", b), ("slope", slope)]:
            expected = 2 * first_grads[param_name]
            torch.testing.assert_close(
                param.grad,
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Gradient accumulation failed for {param_name}",
            )

    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        input_dim, output_dim = 4, 3

        for batch_size in [1, 2, 4, 8]:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, input_dim, requires_grad=True)
                w = torch.randn(output_dim, input_dim, requires_grad=True)
                b = torch.randn(batch_size, output_dim, requires_grad=True)
                slope = torch.randn(1, requires_grad=True)

                # Forward pass
                y = Linear.apply(x, w, b)
                z = LearnedSiLU.apply(y, slope)

                # Check output shape
                self.assertEqual(z.shape, (batch_size, output_dim))

                # Backward pass
                loss = z.sum()
                loss.backward()

                # Check gradient shapes
                self.assertEqual(x.grad.shape, x.shape)
                self.assertEqual(w.grad.shape, w.shape)
                self.assertEqual(b.grad.shape, b.shape)
                self.assertEqual(slope.grad.shape, slope.shape)

    def test_reproducibility(self):
        """Test that results are reproducible with same inputs"""
        # Set random seed
        torch.manual_seed(42)

        x = torch.randn(2, 3, requires_grad=True)
        w = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)
        slope = torch.randn(1, requires_grad=True)

        # First run
        y1 = Linear.apply(x, w, b)
        z1 = LearnedSiLU.apply(y1, slope)
        loss1 = z1.sum()
        loss1.backward()

        grad1 = x.grad.clone()

        # Reset gradients
        x.grad.zero_()

        # Second run with same inputs
        y2 = Linear.apply(x, w, b)
        z2 = LearnedSiLU.apply(y2, slope)
        loss2 = z2.sum()
        loss2.backward()

        # Results should be identical
        torch.testing.assert_close(z1, z2, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(grad1, x.grad, rtol=1e-6, atol=1e-6)

    def test_error_handling(self):
        """Test error handling for mismatched dimensions"""
        # Mismatched dimensions should raise errors
        x = torch.randn(2, 3, requires_grad=True)
        w = torch.randn(4, 5, requires_grad=True)  # Wrong dimensions
        b = torch.randn(2, 4, requires_grad=True)

        with self.assertRaises(RuntimeError):
            Linear.apply(x, w, b)


class TestMainScript(unittest.TestCase):
    """Test the main script functionality"""

    def test_main_script_execution(self):
        """Test that main script runs without errors"""
        # Import and run main script logic
        try:
            # Replicate main.py logic
            x = torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True)
            w = torch.tensor([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]], requires_grad=True)
            b = torch.tensor([[-5.0, -12.0]], requires_grad=True)
            slope = torch.tensor([-1.0], requires_grad=True)

            # Forward pass
            y = Linear.apply(x, w, b)
            z = LearnedSiLU.apply(y, slope)

            # Backward pass
            loss = z.sum()

            # Retain gradients for intermediate tensors
            y.retain_grad()
            z.retain_grad()
            loss.retain_grad()

            loss.backward()

            # Check that computation completed successfully
            self.assertIsNotNone(y)
            self.assertIsNotNone(z)
            self.assertIsNotNone(loss)

            # Check gradients exist
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(w.grad)
            self.assertIsNotNone(b.grad)
            self.assertIsNotNone(slope.grad)

        except Exception as e:
            self.fail(f"Main script execution failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
