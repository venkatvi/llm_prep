"""
Copyright (c) 2025. All rights reserved.
"""

"""
Utility functions for regression models.

This module provides helper functions for visualization and weight initialization
for regression models. Includes plotting utilities and neural network weight
initialization strategies.
"""

import torch 
import matplotlib.pyplot as plt
from logger import Logger

def plot_results(inputs: torch.Tensor, targets: torch.Tensor, y_hat: torch.Tensor, tensorboard_log_dir: str, run_name: str) -> None:
    """
    Create scatter plot comparing actual targets vs model predictions and log to TensorBoard.
    
    This function generates a visualization showing the model's predictions against the true target
    values using different colored scatter plots. The plot is logged to TensorBoard for monitoring
    training performance and model accuracy.
    
    Args:
        inputs (torch.Tensor): Input features used for prediction
        targets (torch.Tensor): True target values from the dataset
        y_hat (torch.Tensor): Model predictions/estimates
        tensorboard_log_dir (str): Directory path for TensorBoard logs
        run_name (str): Name identifier for this training run
    
    Returns:
        None
    """
    # Create matplotlib figure and axis
    fig, ax = plt.subplots()
    
    # Plot true targets in red circles
    ax.scatter(inputs.numpy(), targets.numpy(), color="red", marker="o", label="targets")
    
    # Plot model predictions in blue stars
    ax.scatter(inputs.numpy(), y_hat, color="blue", marker="*", label="y_hat" )
    
    # Configure plot appearance and labels
    ax.set_xlabel("inputs")
    ax.set_ylabel("outputs")
    ax.legend()
    ax.set_title("Scatter plot comparison of Y vs Y^Hat")
    
    # Log the figure to TensorBoard
    logger = Logger(log_dir=tensorboard_log_dir, run_name=run_name)
    logger.log_figure("plots/"+run_name, fig)
    logger.close()
    
    # Close the figure to free memory
    plt.close(fig)

def init_weights(layer: torch.nn.Module) -> None:
    """
    Initialize weights for neural network layers using Kaiming uniform initialization.
    
    This function applies Kaiming uniform initialization to Linear layers, which is
    particularly effective for layers followed by ReLU activations. Biases are
    initialized to zero.
    
    Args:
        layer (torch.nn.Module): Neural network layer to initialize
                                Must be applied to each layer individually (use with model.apply())
    
    Returns:
        None
        
    Example:
        model.apply(init_weights)  # Apply to all layers in a model
    """
    if isinstance(layer, torch.nn.Linear): 
        # Use Kaiming uniform initialization for weights (good for ReLU networks)
        torch.nn.init.kaiming_uniform(layer.weight, nonlinearity="relu")
        
        # Initialize bias to zero if it exists
        if layer.bias is not None: 
            torch.nn.init.zeros_(layer.bias)
