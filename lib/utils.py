"""
Utility functions: plotting and weight initialization.
"""

import torch
import matplotlib.pyplot as plt
from lib.logger import Logger

def plot_results(inputs: torch.Tensor, targets: torch.Tensor, y_hat: torch.Tensor, tensorboard_log_dir: str, run_name: str) -> None:
    """Create scatter plot and log to TensorBoard."""
    fig, ax = plt.subplots()
    ax.scatter(inputs.numpy(), targets.numpy(), color="red", marker="o", label="targets")
    ax.scatter(inputs.numpy(), y_hat, color="blue", marker="*", label="y_hat" )
    ax.set_xlabel("inputs")
    ax.set_ylabel("outputs")
    ax.legend()
    ax.set_title("Scatter plot comparison of Y vs Y^Hat")
    
    logger = Logger(log_dir=tensorboard_log_dir, run_name=run_name)
    logger.log_figure("plots/"+run_name, fig)
    logger.close()
    plt.close(fig)

def init_weights(layer: torch.nn.Module) -> None:
    """Initialize weights using Kaiming uniform for Linear layers."""
    if isinstance(layer, torch.nn.Linear): 
        # Use Kaiming uniform initialization for weights (good for ReLU networks)
        torch.nn.init.kaiming_uniform(layer.weight, nonlinearity="relu")
        
        # Initialize bias to zero if it exists
        if layer.bias is not None: 
            torch.nn.init.zeros_(layer.bias)
