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
Activation function utilities for neural networks.

This module provides a factory function to create various PyTorch activation
layers including ReLU, Tanh, Sigmoid, LeakyReLU, GELU, and SiLU. Used by
regression models to add non-linearity.
"""

import torch 

def get_activation_layer(custom_act: str)->torch.nn.Module: 
    if custom_act == "relu": 
        return torch.nn.ReLU() 
    elif custom_act == "tanh": 
        return torch.nn.Tanh()
    elif custom_act == "sigmoid":
        return torch.nn.Sigmoid()
    elif custom_act == "leakyrelu":
        return torch.nn.LeakyReLU()
    elif custom_act == "gelu":
        return torch.nn.GELU()
    elif custom_act == "silu": 
        return torch.nn.SiLU()
    else:
        raise ValueError("Unsupported activation layer")