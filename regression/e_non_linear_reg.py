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
Non-linear regression model implementation.

This module implements a Multi-Layer Perceptron (MLP) for non-linear regression
using PyTorch. The model can learn complex non-linear relationships in data.
"""

import torch 
from typing import Tuple

class MLP(torch.nn.Module): 
    def __init__(self, hidden_dim: int): 
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_1 = torch.nn.Linear(1, self.hidden_dim)
        self.relu = torch.nn.ReLU() 
        self.linear_2 = torch.nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.linear_1(x)
        x = self.relu(x)
        return self.linear_2(x)

    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]: 
        # Define data 
        inputs = torch.rand(100, 1) * 10
        targets = 4 * inputs**2 + 2 * inputs + torch.rand(100, 1)
        return inputs, targets