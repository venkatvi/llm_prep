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
Utility functions for regression models.

This module provides helper functions for visualization and weight initialization
for regression models. Includes plotting utilities and neural network weight
initialization strategies.
"""

import torch 
import matplotlib.pyplot as plt

def plot_results(inputs: torch.Tensor, targets: torch.Tensor, y_hat: torch.Tensor) -> None: 
    plt.scatter(inputs.numpy(), targets.numpy(), color="red", marker="o", label="targets")
    plt.scatter(inputs.numpy(), y_hat, color="blue", marker="*", label="y_hat" )
    plt.xlabel("inputs")
    plt.ylabel("outputs")
    plt.legend()
    plt.show()

def init_weights(layer: torch.nn.Module): 
    if isinstance(layer, torch.nn.Linear): 
        torch.nn.init.kaiming_uniform(layer.weight, nonlinearity="relu")
        if layer.bias is not None: 
            torch.nn.init.zeros_(layer.bias)
