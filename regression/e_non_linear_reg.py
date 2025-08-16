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
from activations import get_activation_layer

class MLP(torch.nn.Module): 
    def __init__(self, num_latent_layers: int, latent_dim: list[int], custom_act: str, allow_residual: bool=False): 
        super().__init__()
        self.hidden_dim = latent_dim
        self.allow_residual = allow_residual
        self.layers = torch.nn.ModuleList()
        assert len(latent_dim) == num_latent_layers
        for layer_index in range(num_latent_layers): 
            input_dim = 1 if layer_index == 0 else output_dim
            output_dim = latent_dim[layer_index]
            linear_layer = torch.nn.Linear(input_dim, output_dim)
            self.layers.append(linear_layer)

            act_layer = get_activation_layer(custom_act)
            self.layers.append(act_layer)
        self.output_layer = torch.nn.Linear(output_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        for layer in self.layers:
            if not isinstance(layer, torch.nn.Linear) and self.allow_residual: 
                x = layer(x) + x
            else:
                x = layer(x)
        return self.output_layer(x)

    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]: 
        # Define data 
        inputs = torch.rand(100, 1) * 10
        targets = 4 * inputs**2 + 2 * inputs + torch.rand(100, 1)
        return inputs, targets