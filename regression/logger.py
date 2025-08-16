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
TensorBoard logging utilities for regression models.

This module provides a Logger class that wraps PyTorch's SummaryWriter to
provide convenient logging of training metrics to TensorBoard. Supports
scalar logging with both TensorBoard visualization and console output.
"""

from torch.utils.tensorboard import SummaryWriter
from typing import Union

class Logger: 
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
    
    def log_scalars(self, scalar_dict: dict[str, Union[int, float]]) -> None: 
        self.writer.add_scalars('Update', scalar_dict)
        string_to_print = ""
        for k, v in scalar_dict.items():
            string_to_print += "".join(k, ":", v, ",")
        print(string_to_print)


    