"""
Activation functions: Tanh, Sigmoid, ReLU, LearnedSiLU
"""

from typing import Tuple

import torch

from simple import Exp

class Tanh(torch.autograd.Function): 
    """Hyperbolic tangent: tanh(x) with gradient 1 - tanh²(x)"""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor: 
        ctx.save_for_backward(input)
        return torch.tanh(input)
    
    @staticmethod 
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input, ) = ctx.saved_tensors 
        tanh_val = torch.tanh(input)
        return grad_output * (1 - tanh_val ** 2)

class Sigmoid(torch.autograd.Function): 
    """Sigmoid: 1/(1 + e^(-x)) with gradient σ(x)(1 - σ(x))"""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor: 
        ctx.save_for_backward(input)
        return 1 / (1.0 + Exp.apply(-input))
    
    @staticmethod 
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input, ) = ctx.saved_tensors 
        sigmoid = 1 / (1.0 + Exp.apply(-input))
        return grad_output * sigmoid * (1 - sigmoid) 
    
class ReLU(torch.autograd.Function):
    """ReLU: max(0, x) with gradient 1 if x > 0, else 0"""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor: 
        ctx.save_for_backward(input)
        return torch.clamp(input, min=0)
    
    @staticmethod 
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input, ) = ctx.saved_tensors 
        return grad_output * (input > 0).float() 
    
class LearnedSiLU(torch.autograd.Function):
    """Learnable SiLU: α × x × σ(x) with trainable slope α"""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input, slope)
        sigmoid_x = Sigmoid.apply(input)
        return slope * input * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input, slope = ctx.saved_tensors 
        
        sigmoid_x = Sigmoid.apply(input)
        sigmoid_prime = sigmoid_x * (1 - sigmoid_x)
        
        grad_input = grad_output * slope * (sigmoid_x + input * sigmoid_prime)
        grad_slope = (grad_output * input * sigmoid_x).sum(dim=0, keepdim=True)
        
        return grad_input, grad_slope  
       
        