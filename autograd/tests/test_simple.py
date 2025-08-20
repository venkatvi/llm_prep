"""
Copyright (c) 2025. All rights reserved.
"""

"""
Tests for simple mathematical functions in autograd.

This module tests the basic mathematical operations: Power, Square, Cube, and Exp.
Tests verify forward computation, gradient correctness, and edge cases.
"""

import torch
import unittest
import math
import sys
import os

# Add parent directory to path to import autograd modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple import Power, Square, Cube, Exp


class TestPower(unittest.TestCase):
    """Test cases for Power function: f(x, n) = x^n"""
    
    def test_forward_basic(self):
        """Test basic forward computation"""
        x = torch.tensor(2.0)
        n = torch.tensor(3.0)
        result = Power.apply(x, n)
        expected = 8.0
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_forward_fractional_power(self):
        """Test forward with fractional exponent"""
        x = torch.tensor(4.0)
        n = torch.tensor(0.5)
        result = Power.apply(x, n)
        expected = 2.0  # sqrt(4) = 2
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_gradients(self):
        """Test gradient computation for both inputs"""
        x = torch.tensor(2.0, requires_grad=True)
        n = torch.tensor(3.0, requires_grad=True)
        
        y = Power.apply(x, n)
        y.backward()
        
        # Expected gradients:
        # ∂y/∂x = n * x^(n-1) = 3 * 2^2 = 12
        # ∂y/∂n = x^n * ln(x) = 8 * ln(2) ≈ 5.5452
        expected_grad_x = 12.0
        expected_grad_n = 8.0 * math.log(2.0)
        
        self.assertAlmostEqual(x.grad.item(), expected_grad_x, places=4)
        self.assertAlmostEqual(n.grad.item(), expected_grad_n, places=4)
    
    def test_gradient_check(self):
        """Test gradients using PyTorch's gradient checker"""
        def power_func(x, n):
            return Power.apply(x, n)
        
        x = torch.randn(1, dtype=torch.double, requires_grad=True)
        n = torch.randn(1, dtype=torch.double, requires_grad=True)
        
        # Make sure x is positive for log computation
        x.data = torch.abs(x.data) + 0.1
        
        test_result = torch.autograd.gradcheck(power_func, (x, n), eps=1e-6, atol=1e-4)
        self.assertTrue(test_result)


class TestSquare(unittest.TestCase):
    """Test cases for Square function: f(x) = x^2"""
    
    def test_forward_positive(self):
        """Test forward with positive input"""
        x = torch.tensor(3.0)
        result = Square.apply(x)
        expected = 9.0
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_forward_negative(self):
        """Test forward with negative input"""
        x = torch.tensor(-4.0)
        result = Square.apply(x)
        expected = 16.0
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_forward_zero(self):
        """Test forward with zero input"""
        x = torch.tensor(0.0)
        result = Square.apply(x)
        expected = 0.0
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_gradient(self):
        """Test gradient computation: df/dx = 2x"""
        x = torch.tensor(5.0, requires_grad=True)
        y = Square.apply(x)
        y.backward()
        
        expected_grad = 10.0  # 2 * 5
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)
    
    def test_gradient_negative(self):
        """Test gradient with negative input"""
        x = torch.tensor(-3.0, requires_grad=True)
        y = Square.apply(x)
        y.backward()
        
        expected_grad = -6.0  # 2 * (-3)
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)
    
    def test_gradient_check(self):
        """Test gradients using PyTorch's gradient checker"""
        x = torch.randn(5, dtype=torch.double, requires_grad=True)
        test_result = torch.autograd.gradcheck(Square.apply, x, eps=1e-6, atol=1e-4)
        self.assertTrue(test_result)


class TestCube(unittest.TestCase):
    """Test cases for Cube function: f(x) = x^3"""
    
    def test_forward_positive(self):
        """Test forward with positive input"""
        x = torch.tensor(2.0)
        result = Cube.apply(x)
        expected = 8.0
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_forward_negative(self):
        """Test forward with negative input"""
        x = torch.tensor(-3.0)
        result = Cube.apply(x)
        expected = -27.0
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_gradient(self):
        """Test gradient computation: df/dx = 3x^2"""
        x = torch.tensor(4.0, requires_grad=True)
        y = Cube.apply(x)
        y.backward()
        
        expected_grad = 48.0  # 3 * 4^2
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)
    
    def test_gradient_negative(self):
        """Test gradient with negative input"""
        x = torch.tensor(-2.0, requires_grad=True)
        y = Cube.apply(x)
        y.backward()
        
        expected_grad = 12.0  # 3 * (-2)^2 = 3 * 4
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)
    
    def test_gradient_check(self):
        """Test gradients using PyTorch's gradient checker"""
        x = torch.randn(3, dtype=torch.double, requires_grad=True)
        test_result = torch.autograd.gradcheck(Cube.apply, x, eps=1e-6, atol=1e-4)
        self.assertTrue(test_result)


class TestExp(unittest.TestCase):
    """Test cases for Exp function: f(x) = e^x"""
    
    def test_forward_zero(self):
        """Test forward with zero input"""
        x = torch.tensor(0.0)
        result = Exp.apply(x)
        expected = 1.0
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_forward_positive(self):
        """Test forward with positive input"""
        x = torch.tensor(1.0)
        result = Exp.apply(x)
        expected = math.e
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_forward_negative(self):
        """Test forward with negative input"""
        x = torch.tensor(-1.0)
        result = Exp.apply(x)
        expected = 1.0 / math.e
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_gradient(self):
        """Test gradient computation: df/dx = e^x"""
        x = torch.tensor(2.0, requires_grad=True)
        y = Exp.apply(x)
        y.backward()
        
        expected_grad = math.exp(2.0)  # e^2
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)
    
    def test_gradient_at_zero(self):
        """Test gradient at zero"""
        x = torch.tensor(0.0, requires_grad=True)
        y = Exp.apply(x)
        y.backward()
        
        expected_grad = 1.0  # e^0
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=4)
    
    def test_gradient_check(self):
        """Test gradients using PyTorch's gradient checker"""
        x = torch.randn(4, dtype=torch.double, requires_grad=True)
        test_result = torch.autograd.gradcheck(Exp.apply, x, eps=1e-6, atol=1e-4)
        self.assertTrue(test_result)


class TestTensorOperations(unittest.TestCase):
    """Test tensor operations with multiple elements"""
    
    def test_square_tensor(self):
        """Test Square with tensor input"""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Square.apply(x)
        
        expected = torch.tensor([1.0, 4.0, 9.0])
        torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)
        
        # Test gradients
        loss = y.sum()
        loss.backward()
        
        expected_grad = torch.tensor([2.0, 4.0, 6.0])  # 2 * [1, 2, 3]
        torch.testing.assert_close(x.grad, expected_grad, rtol=1e-4, atol=1e-4)
    
    def test_cube_tensor(self):
        """Test Cube with tensor input"""
        x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        y = Cube.apply(x)
        
        expected = torch.tensor([1.0, -8.0, 27.0])
        torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)
        
        # Test gradients
        loss = y.sum()
        loss.backward()
        
        expected_grad = torch.tensor([3.0, 12.0, 27.0])  # 3 * [1^2, (-2)^2, 3^2]
        torch.testing.assert_close(x.grad, expected_grad, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()