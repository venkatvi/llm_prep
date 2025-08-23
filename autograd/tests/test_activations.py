"""
Copyright (c) 2025. All rights reserved.
"""

"""
Tests for activation functions in autograd.

This module tests custom activation functions including Tanh, Sigmoid, ReLU,
and LearnedSiLU. Tests verify forward computation, gradient correctness, and
proper handling of edge cases.
"""

import math
import os
import sys
import unittest

import torch

# Add parent directory to path to import autograd modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activations import LearnedSiLU, ReLU, Sigmoid, Tanh


class TestTanh(unittest.TestCase):
    """Test cases for Tanh function: f(x) = tanh(x)"""

    def test_forward_zero(self):
        """Test forward with zero input"""
        x = torch.tensor(0.0)
        result = Tanh.apply(x)
        expected = 0.0
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_forward_positive(self):
        """Test forward with positive input"""
        x = torch.tensor(1.0)
        result = Tanh.apply(x)
        expected = math.tanh(1.0)
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_forward_negative(self):
        """Test forward with negative input"""
        x = torch.tensor(-2.0)
        result = Tanh.apply(x)
        expected = math.tanh(-2.0)
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_gradient(self):
        """Test gradient computation: df/dx = 1 - tanh^2(x)"""
        x = torch.tensor(0.5, requires_grad=True)
        y = Tanh.apply(x)
        y.backward()

        tanh_val = math.tanh(0.5)
        expected_grad = 1 - tanh_val**2
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)

    def test_gradient_at_zero(self):
        """Test gradient at zero (should be 1)"""
        x = torch.tensor(0.0, requires_grad=True)
        y = Tanh.apply(x)
        y.backward()

        expected_grad = 1.0  # 1 - tanh^2(0) = 1 - 0 = 1
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)


class TestSigmoid(unittest.TestCase):
    """Test cases for Sigmoid function: f(x) = 1/(1 + e^(-x))"""

    def test_forward_zero(self):
        """Test forward with zero input"""
        x = torch.tensor(0.0)
        result = Sigmoid.apply(x)
        expected = 0.5
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_forward_positive(self):
        """Test forward with positive input"""
        x = torch.tensor(2.0)
        result = Sigmoid.apply(x)
        expected = 1.0 / (1.0 + math.exp(-2.0))
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_forward_negative(self):
        """Test forward with negative input"""
        x = torch.tensor(-1.0)
        result = Sigmoid.apply(x)
        expected = 1.0 / (1.0 + math.exp(1.0))
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_gradient(self):
        """Test gradient computation: df/dx = sigmoid(x) * (1 - sigmoid(x))"""
        x = torch.tensor(1.0, requires_grad=True)
        y = Sigmoid.apply(x)
        y.backward()

        sigmoid_val = 1.0 / (1.0 + math.exp(-1.0))
        expected_grad = sigmoid_val * (1 - sigmoid_val)
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)

    def test_gradient_at_zero(self):
        """Test gradient at zero (should be 0.25)"""
        x = torch.tensor(0.0, requires_grad=True)
        y = Sigmoid.apply(x)
        y.backward()

        expected_grad = 0.25  # 0.5 * (1 - 0.5)
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)

    def test_saturation_positive(self):
        """Test saturation for large positive values"""
        x = torch.tensor(10.0)
        result = Sigmoid.apply(x)
        # Should be close to 1.0
        self.assertGreater(result.item(), 0.99)

    def test_saturation_negative(self):
        """Test saturation for large negative values"""
        x = torch.tensor(-10.0)
        result = Sigmoid.apply(x)
        # Should be close to 0.0
        self.assertLess(result.item(), 0.01)


class TestReLU(unittest.TestCase):
    """Test cases for ReLU function: f(x) = max(0, x)"""

    def test_forward_positive(self):
        """Test forward with positive input"""
        x = torch.tensor(5.0)
        result = ReLU.apply(x)
        expected = 5.0
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_forward_negative(self):
        """Test forward with negative input"""
        x = torch.tensor(-3.0)
        result = ReLU.apply(x)
        expected = 0.0
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_forward_zero(self):
        """Test forward with zero input"""
        x = torch.tensor(0.0)
        result = ReLU.apply(x)
        expected = 0.0
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_gradient_positive(self):
        """Test gradient for positive input (should be 1)"""
        x = torch.tensor(2.0, requires_grad=True)
        y = ReLU.apply(x)
        y.backward()

        expected_grad = 1.0
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)

    def test_gradient_negative(self):
        """Test gradient for negative input (should be 0)"""
        x = torch.tensor(-1.0, requires_grad=True)
        y = ReLU.apply(x)
        y.backward()

        expected_grad = 0.0
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)

    def test_gradient_zero(self):
        """Test gradient at zero (should be 0)"""
        x = torch.tensor(0.0, requires_grad=True)
        y = ReLU.apply(x)
        y.backward()

        expected_grad = 0.0  # By convention, ReLU gradient at 0 is 0
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)

    def test_tensor_input(self):
        """Test ReLU with tensor input"""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = ReLU.apply(x)

        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)

        # Test gradients
        loss = y.sum()
        loss.backward()

        expected_grad = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        torch.testing.assert_close(x.grad, expected_grad, rtol=1e-4, atol=1e-4)


class TestLearnedSiLU(unittest.TestCase):
    """Test cases for LearnedSiLU function: f(x) = slope * x * sigmoid(x)"""

    def test_forward_basic(self):
        """Test basic forward computation"""
        x = torch.tensor(1.0)
        slope = torch.tensor(2.0)
        result = LearnedSiLU.apply(x, slope)

        # Expected: 2 * 1 * sigmoid(1) = 2 * sigmoid(1)
        sigmoid_val = 1.0 / (1.0 + math.exp(-1.0))
        expected = 2.0 * sigmoid_val
        self.assertAlmostEqual(result.item(), expected, places=4)

    def test_forward_zero_input(self):
        """Test forward with zero input"""
        x = torch.tensor(0.0)
        slope = torch.tensor(1.5)
        result = LearnedSiLU.apply(x, slope)

        # Expected: 1.5 * 0 * sigmoid(0) = 0
        expected = 0.0
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_forward_negative_slope(self):
        """Test forward with negative slope"""
        x = torch.tensor(2.0)
        slope = torch.tensor(-1.0)
        result = LearnedSiLU.apply(x, slope)

        sigmoid_val = 1.0 / (1.0 + math.exp(-2.0))
        expected = -1.0 * 2.0 * sigmoid_val
        self.assertAlmostEqual(result.item(), expected, places=4)

    def test_gradients(self):
        """Test gradient computation for both inputs"""
        x = torch.tensor(1.0, requires_grad=True)
        slope = torch.tensor(2.0, requires_grad=True)

        y = LearnedSiLU.apply(x, slope)
        y.backward()

        # Compute expected gradients manually
        sigmoid_val = 1.0 / (1.0 + math.exp(-1.0))
        sigmoid_prime = sigmoid_val * (1 - sigmoid_val)

        # ∂y/∂x = slope * [sigmoid(x) + x * sigmoid'(x)]
        expected_grad_x = 2.0 * (sigmoid_val + 1.0 * sigmoid_prime)

        # ∂y/∂slope = x * sigmoid(x)
        expected_grad_slope = 1.0 * sigmoid_val

        self.assertAlmostEqual(x.grad.item(), expected_grad_x, places=4)
        self.assertAlmostEqual(slope.grad.item(), expected_grad_slope, places=4)

    def test_gradients_zero_input(self):
        """Test gradients when input is zero"""
        x = torch.tensor(0.0, requires_grad=True)
        slope = torch.tensor(3.0, requires_grad=True)

        y = LearnedSiLU.apply(x, slope)
        y.backward()

        # At x=0: sigmoid(0) = 0.5, sigmoid'(0) = 0.25
        # ∂y/∂x = slope * [0.5 + 0 * 0.25] = 3 * 0.5 = 1.5
        # ∂y/∂slope = 0 * 0.5 = 0

        expected_grad_x = 1.5
        expected_grad_slope = 0.0

        self.assertAlmostEqual(x.grad.item(), expected_grad_x, places=4)
        self.assertAlmostEqual(slope.grad.item(), expected_grad_slope, places=4)

    def test_tensor_inputs(self):
        """Test with tensor inputs"""
        x = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
        slope = torch.tensor([1.0], requires_grad=True)  # Broadcast slope

        y = LearnedSiLU.apply(x, slope)

        # Verify shape
        self.assertEqual(y.shape, x.shape)

        # Test backward
        loss = y.sum()
        loss.backward()

        # Check gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(slope.grad.shape, slope.shape)

    def test_gradient_check(self):
        """Test gradients using PyTorch's gradient checker"""

        def learned_silu_func(x, slope):
            return LearnedSiLU.apply(x, slope)

        x = torch.randn(1, dtype=torch.double, requires_grad=True)
        slope = torch.randn(1, dtype=torch.double, requires_grad=True)

        test_result = torch.autograd.gradcheck(learned_silu_func, (x, slope), eps=1e-6, atol=1e-4)
        self.assertTrue(test_result)


class TestActivationConsistency(unittest.TestCase):
    """Test consistency between custom and PyTorch implementations"""

    def test_relu_consistency(self):
        """Compare custom ReLU with PyTorch ReLU"""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Custom implementation
        y_custom = ReLU.apply(x)

        # PyTorch implementation
        y_pytorch = torch.relu(x)

        torch.testing.assert_close(y_custom, y_pytorch, rtol=1e-4, atol=1e-4)

    def test_sigmoid_consistency(self):
        """Compare custom Sigmoid with PyTorch Sigmoid"""
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

        # Custom implementation
        y_custom = Sigmoid.apply(x)

        # PyTorch implementation
        y_pytorch = torch.sigmoid(x)

        torch.testing.assert_close(y_custom, y_pytorch, rtol=1e-3, atol=1e-3)

    def test_tanh_consistency(self):
        """Compare custom Tanh with PyTorch Tanh"""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Custom implementation
        y_custom = Tanh.apply(x)

        # PyTorch implementation
        y_pytorch = torch.tanh(x)

        torch.testing.assert_close(y_custom, y_pytorch, rtol=1e-4, atol=1e-4)


class TestActivationEdgeCases(unittest.TestCase):
    """Test edge cases for activation functions"""

    def test_large_values(self):
        """Test activation functions with large input values"""
        large_pos = torch.tensor(100.0)
        large_neg = torch.tensor(-100.0)

        # ReLU should handle large values fine
        relu_pos = ReLU.apply(large_pos)
        relu_neg = ReLU.apply(large_neg)
        self.assertEqual(relu_pos.item(), 100.0)
        self.assertEqual(relu_neg.item(), 0.0)

        # Sigmoid should saturate properly
        sig_pos = Sigmoid.apply(large_pos)
        sig_neg = Sigmoid.apply(large_neg)
        self.assertAlmostEqual(sig_pos.item(), 1.0, places=5)
        self.assertAlmostEqual(sig_neg.item(), 0.0, places=5)

    def test_very_small_values(self):
        """Test activation functions with very small input values"""
        small_val = torch.tensor(1e-8)

        # All functions should handle small values
        relu_result = ReLU.apply(small_val)
        sigmoid_result = Sigmoid.apply(small_val)
        tanh_result = Tanh.apply(small_val)

        self.assertAlmostEqual(relu_result.item(), 1e-8, places=10)
        self.assertAlmostEqual(sigmoid_result.item(), 0.5 + 1e-8 / 4, places=8)  # Linear approx
        self.assertAlmostEqual(tanh_result.item(), 1e-8, places=10)  # Linear approx


if __name__ == "__main__":
    unittest.main()
