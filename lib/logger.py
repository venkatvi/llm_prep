"""
Copyright (c) 2025. All rights reserved.
"""

"""
TensorBoard logging utilities for regression models.

This module provides a Logger class that wraps PyTorch's SummaryWriter to
provide convenient logging of training metrics to TensorBoard. Supports
scalar logging with both TensorBoard visualization and console output.
"""

import torch
from typing import Union

from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter
class Logger:
    """
    TensorBoard logging wrapper for regression model training.
    
    Provides convenient methods to log scalars, tensors, and matplotlib figures
    to TensorBoard while also providing console output for immediate feedback.
    """
    
    def __init__(self, log_dir: str, run_name: str = None):
        """
        Initialize the Logger with TensorBoard SummaryWriter.
        
        Args:
            log_dir (str): Base directory for TensorBoard logs
            run_name (str, optional): Name for this specific run. Creates subdirectory if provided.
        """
        if run_name:
            self.log_dir = f"{log_dir}/{run_name}"
        else:
            self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)
       
    def log_scalars(self, scalar_dict: dict[str, Union[int, float]], step: int = 0) -> None:
        """
        Log scalar values to TensorBoard and print to console.
        
        Args:
            scalar_dict (dict): Dictionary of metric names and values
            step (int): Global step number for TensorBoard timeline
        """
        for k, v in scalar_dict.items():
            self.writer.add_scalar(k, v, step)
        
        # Console output
        string_to_print = ""
        for k, v in scalar_dict.items():
            string_to_print += f"{k}:{v}, "
        print(string_to_print.rstrip(", "))

    def log_tensor(self, tensor_dict: dict[str, torch.Tensor]) -> None:
        """
        Log tensor values as scalars to TensorBoard.
        
        Args:
            tensor_dict (dict): Dictionary of tensor names and PyTorch tensors
                              Tensors are converted to scalar values using .item()
        """
        for k, v in tensor_dict.items(): 
            self.writer.add_scalar(k, v.item())

    def log_figure(self, tag: str, fig: Figure, step: int = 0) -> None:
        """
        Log matplotlib figure to TensorBoard.
        
        Args:
            tag (str): Name/tag for the figure in TensorBoard
            fig (Figure): Matplotlib figure object to log
            step (int): Global step number for TensorBoard timeline
        """
        # Use TensorBoard's built-in add_figure method
        self.writer.add_figure(tag, fig, step)

    def close(self) -> None:
        """
        Close the TensorBoard writer and flush any remaining data.
        Should be called when logging is complete to ensure all data is written.
        """
        self.writer.close()
    