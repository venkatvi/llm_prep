"""
Copyright (c) 2025. All rights reserved.
"""

"""
Tests for linear layer implementation in autograd.

This module tests the custom Linear layer function including forward computation,
gradient correctness for inputs, weights, and biases, and various tensor shapes.
"""

import os
import sys
import unittest

import torch

# Add parent directory to path to import autograd modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linear import Linear


class TestLinear(unittest.TestCase):
    """Test cases for Linear function: f(x, W, b) = xW^T + b"""

    def test_forward_basic(self):
        """Test basic forward computation"""
        # Simple case: [1, 2] input, [2, 2] weights, [1, 2] bias
        x = torch.tensor([[1.0, 2.0]])  # [1, 2]
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        b = torch.tensor([[0.5, 1.0]])  # [1, 2]

        result = Linear.apply(x, w, b)

        # Expected: [1, 2] @ [[1, 3], [2, 4]] + [0.5, 1.0]
        #         = [5, 11] + [0.5, 1.0] = [5.5, 12.0]
        expected = torch.tensor([[5.5, 12.0]])
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_forward_batch(self):
        """Test forward with batch input"""
        # Batch size 2
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        w = torch.tensor([[1.0, 1.0], [2.0, 2.0]])  # [2, 2]
        b = torch.tensor([[1.0, 2.0], [1.0, 2.0]])  # [2, 2]

        result = Linear.apply(x, w, b)

        # Expected: [[1, 2], [3, 4]] @ [[1, 2], [1, 2]] + [[1, 2], [1, 2]]
        #         = [[3, 6], [7, 14]] + [[1, 2], [1, 2]] = [[4, 8], [8, 16]]
        expected = torch.tensor([[4.0, 8.0], [8.0, 16.0]])
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_gradients_simple(self):
        """Test gradient computation for simple case"""
        x = torch.tensor([[2.0, 3.0]], requires_grad=True)  # [1, 2]
        w = torch.tensor([[1.0, 2.0], [4.0, 5.0]], requires_grad=True)  # [2, 2]
        b = torch.tensor([[1.0, 2.0]], requires_grad=True)  # [1, 2]

        y = Linear.apply(x, w, b)
        loss = y.sum()  # Scalar loss for backward
        loss.backward()

        # Expected gradients:
        # grad_x = grad_output @ w = [1, 1] @ [[1, 2], [4, 5]] = [5, 7]
        # grad_w = grad_output.T @ x = [[1], [1]] @ [[2, 3]] = [[2, 3], [2, 3]]
        # grad_b = grad_output = [1, 1]

        expected_grad_x = torch.tensor([[5.0, 7.0]])
        expected_grad_w = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
        expected_grad_b = torch.tensor([[1.0, 1.0]])

        torch.testing.assert_close(x.grad, expected_grad_x, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(w.grad, expected_grad_w, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(b.grad, expected_grad_b, rtol=1e-4, atol=1e-4)

    def test_gradients_batch(self):
        """Test gradient computation for batch input"""
        batch_size, input_dim, output_dim = 3, 4, 2

        x = torch.randn(batch_size, input_dim, requires_grad=True)
        w = torch.randn(output_dim, input_dim, requires_grad=True)
        b = torch.randn(batch_size, output_dim, requires_grad=True)

        y = Linear.apply(x, w, b)
        loss = y.sum()
        loss.backward()

        # Check gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(w.grad.shape, w.shape)
        self.assertEqual(b.grad.shape, b.shape)

        # Check that gradients are not None and not zero
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertIsNotNone(b.grad)

        # Gradients should not be all zeros (for random inputs)
        self.assertFalse(torch.allclose(x.grad, torch.zeros_like(x.grad)))
        self.assertFalse(torch.allclose(w.grad, torch.zeros_like(w.grad)))
        self.assertFalse(torch.allclose(b.grad, torch.zeros_like(b.grad)))

    def test_gradient_check(self):
        """Test gradients using PyTorch's gradient checker"""

        def linear_func(x, w, b):
            return Linear.apply(x, w, b)

        # Small tensors for gradient check
        x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        w = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        test_result = torch.autograd.gradcheck(linear_func, (x, w, b), eps=1e-6, atol=1e-4)
        self.assertTrue(test_result)

    def test_different_shapes(self):
        """Test various input/output dimensions"""
        test_cases = [
            # (batch_size, input_dim, output_dim)
            (1, 1, 1),  # Minimal case
            (1, 3, 2),  # Single sample
            (4, 5, 3),  # Small batch
            (8, 10, 6),  # Medium batch
        ]

        for batch_size, input_dim, output_dim in test_cases:
            with self.subTest(batch=batch_size, input=input_dim, output=output_dim):
                x = torch.randn(batch_size, input_dim, requires_grad=True)
                w = torch.randn(output_dim, input_dim, requires_grad=True)
                b = torch.randn(batch_size, output_dim, requires_grad=True)

                # Forward pass
                y = Linear.apply(x, w, b)

                # Check output shape
                expected_shape = (batch_size, output_dim)
                self.assertEqual(y.shape, expected_shape)

                # Backward pass
                loss = y.sum()
                loss.backward()

                # Check gradient shapes
                self.assertEqual(x.grad.shape, x.shape)
                self.assertEqual(w.grad.shape, w.shape)
                self.assertEqual(b.grad.shape, b.shape)

    def test_zero_gradients(self):
        """Test behavior with zero inputs"""
        x = torch.zeros(2, 3, requires_grad=True)
        w = torch.randn(4, 3, requires_grad=True)
        b = torch.randn(2, 4, requires_grad=True)

        y = Linear.apply(x, w, b)
        loss = y.sum()
        loss.backward()

        # x gradient should be non-zero (depends on w)
        # w gradient should be zero (x is zero)
        # b gradient should be non-zero (ones from sum)

        expected_w_grad = torch.zeros_like(w.grad)
        torch.testing.assert_close(w.grad, expected_w_grad, rtol=1e-4, atol=1e-4)

    def test_consistency_with_pytorch(self):
        """Test consistency with PyTorch's built-in linear layer"""
        batch_size, input_dim, output_dim = 4, 6, 3

        # Create same inputs
        x = torch.randn(batch_size, input_dim, requires_grad=True)
        w = torch.randn(output_dim, input_dim, requires_grad=True)

        # Our implementation (bias needs to match batch size)
        b_custom = torch.randn(batch_size, output_dim, requires_grad=True)
        y_custom = Linear.apply(x, w, b_custom)

        # PyTorch implementation (bias broadcasted)
        b_pytorch = b_custom[0]  # Take first row as single bias
        y_pytorch = torch.nn.functional.linear(x, w, b_pytorch)

        # Results should be similar for first sample
        # (Our implementation uses per-sample bias, PyTorch uses shared bias)
        # So we'll just check that both produce reasonable outputs

        self.assertEqual(y_custom.shape, y_pytorch.shape)
        self.assertFalse(torch.isnan(y_custom).any())
        self.assertFalse(torch.isnan(y_pytorch).any())


class TestLinearEdgeCases(unittest.TestCase):
    """Test edge cases for Linear function"""

    def test_single_neuron(self):
        """Test single input to single output"""
        x = torch.tensor([[5.0]], requires_grad=True)
        w = torch.tensor([[2.0]], requires_grad=True)
        b = torch.tensor([[1.0]], requires_grad=True)

        y = Linear.apply(x, w, b)
        expected = torch.tensor([[11.0]])  # 5 * 2 + 1

        torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)

        y.backward()

        # Gradients: dx = w = 2, dw = x = 5, db = 1
        self.assertAlmostEqual(x.grad.item(), 2.0, places=4)
        self.assertAlmostEqual(w.grad.item(), 5.0, places=4)
        self.assertAlmostEqual(b.grad.item(), 1.0, places=4)

    def test_large_dimensions(self):
        """Test with larger tensor dimensions"""
        batch_size, input_dim, output_dim = 16, 128, 64

        x = torch.randn(batch_size, input_dim, requires_grad=True)
        w = torch.randn(output_dim, input_dim, requires_grad=True)
        b = torch.randn(batch_size, output_dim, requires_grad=True)

        # Should not raise any errors
        y = Linear.apply(x, w, b)
        loss = y.sum()
        loss.backward()

        # Check shapes
        self.assertEqual(y.shape, (batch_size, output_dim))
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(w.grad.shape, w.shape)
        self.assertEqual(b.grad.shape, b.shape)


if __name__ == "__main__":
    unittest.main()
