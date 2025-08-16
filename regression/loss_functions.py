"""
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Loss function utilities for regression models.

This module provides custom loss functions and a factory function to create
various PyTorch loss functions. Includes MSE, Huber loss, and CrossEntropy
for different training scenarios.
"""

import torch 

class HuberLoss(torch.nn.Module): 
    def __init__(self, delta: float = 1):
        super().__init__()
        self.delta = delta 

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        l2_loss = 0.5* (y_hat - y)**2
        l1_loss = torch.abs(y_hat - y) 
        loss = torch.where(
            l1_loss < self.delta, 
            l2_loss, 
            self.delta * (l1_loss - 0.5)
        )
        return loss.mean() # single value 

def get_loss_function(custom_loss: str) -> torch.nn.Module: 
    if custom_loss == "mse":
        return torch.nn.MSELoss()
    elif custom_loss == "huber":
        return HuberLoss()
    elif custom_loss == "crossentropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss function")