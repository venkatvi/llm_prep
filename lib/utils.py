"""
Utility functions: plotting and weight initialization.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.logger import Logger


def plot_results(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    y_hat: torch.Tensor,
    tensorboard_log_dir: str,
    run_name: str,
) -> None:
    """Create scatter plot comparing predictions vs ground truth and log to TensorBoard.

    Generates a scatter plot visualization showing both ground truth targets and model
    predictions against the input values. Useful for visualizing model performance
    and identifying patterns in prediction errors.

    Args:
        inputs (torch.Tensor): Input values for x-axis [num_samples, 1]
        targets (torch.Tensor): Ground truth target values [num_samples, 1]
        y_hat (torch.Tensor): Model predictions [num_samples, 1]
        tensorboard_log_dir (str): Directory for TensorBoard logs
        run_name (str): Name for this visualization run
    """
    fig, ax = plt.subplots()
    ax.scatter(inputs.numpy(), targets.numpy(), color="red", marker="o", label="targets")
    ax.scatter(inputs.numpy(), y_hat, color="blue", marker="*", label="y_hat")
    ax.set_xlabel("inputs")
    ax.set_ylabel("outputs")
    ax.legend()
    ax.set_title("Scatter plot comparison of Y vs Y^Hat")

    logger = Logger(log_dir=tensorboard_log_dir, run_name=run_name)
    logger.log_figure("plots/" + run_name, fig)
    logger.close()
    plt.close(fig)


def init_weights(layer: torch.nn.Module) -> None:
    """Initialize weights using Kaiming uniform initialization for Linear layers.

    Applies Kaiming uniform initialization to Linear layer weights, which is
    particularly suitable for networks using ReLU activation functions. Sets
    bias terms to zero when present. This function can be used with
    model.apply(init_weights) to initialize an entire network.

    Args:
        layer (torch.nn.Module): Neural network layer to initialize
                               Only Linear layers are affected, others are ignored

    Example:
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        model.apply(init_weights)  # Initialize all Linear layers
    """
    if isinstance(layer, torch.nn.Linear):
        # Use Kaiming uniform initialization for weights (good for ReLU networks)
        torch.nn.init.kaiming_uniform(layer.weight, nonlinearity="relu")

        # Initialize bias to zero if it exists
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


def set_seed(random_seed: int) -> None:
    """Set random seeds for reproducible experiments.

    Sets the random seed for Python's random module, PyTorch, and NumPy to ensure
    reproducible results across different runs. This is essential for debugging
    and comparing different model architectures or hyperparameters fairly.

    Args:
        random_seed (int): Seed value to use for all random number generators

    Example:
        set_seed(42)  # Ensures reproducible results
        data = torch.rand(100, 10)  # Will generate same data every time
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
