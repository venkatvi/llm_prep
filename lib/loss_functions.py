"""
Loss function utilities: MSE, Huber, CrossEntropy.
"""

import torch


class HuberLoss(torch.nn.Module):
    """Huber loss: robust regression loss function, less sensitive to outliers.

    The Huber loss combines the best properties of Mean Squared Error (MSE) and
    Mean Absolute Error (MAE). It uses squared error for small errors (providing
    smooth gradients) and absolute error for large errors (reducing sensitivity
    to outliers).

    The loss function is defined as:
    - If |error| <= delta: loss = 0.5 * error^2
    - If |error| > delta: loss = delta * (|error| - 0.5 * delta)

    Attributes:
        delta (float): Threshold parameter that determines the transition point
                      between quadratic and linear loss
    """

    def __init__(self, delta: float = 1) -> None:
        """Initialize Huber loss with threshold parameter.

        Args:
            delta (float, optional): Threshold for switching from quadratic to linear loss.
                                   Defaults to 1.0. Smaller values make the loss more robust
                                   to outliers but may slow convergence.
        """
        super().__init__()
        self.delta = delta

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss between predictions and targets.

        Args:
            y (torch.Tensor): Ground truth target values
            y_hat (torch.Tensor): Predicted values from the model

        Returns:
            torch.Tensor: Scalar Huber loss value averaged over all samples
        """
        l2_loss = 0.5 * (y_hat - y) ** 2
        l1_loss = torch.abs(y_hat - y)
        loss = torch.where(l1_loss < self.delta, l2_loss, self.delta * (l1_loss - 0.5 * self.delta))
        return loss.mean()


def get_loss_function(custom_loss: str) -> torch.nn.Module:
    """Factory function for creating loss function instances.

    Creates and returns appropriate loss function based on string identifier.
    Supports common loss functions for regression and classification tasks.

    Args:
        custom_loss (str): Loss function identifier. Supported values:
                          - 'mse': Mean Squared Error for regression
                          - 'huber': Huber loss for robust regression
                          - 'crossentropy': Cross Entropy for classification

    Returns:
        torch.nn.Module: Instantiated loss function ready for training

    Raises:
        ValueError: If unsupported loss function identifier is provided

    Example:
        loss_fn = get_loss_function('mse')
        loss = loss_fn(predictions, targets)
    """
    if custom_loss == "mse":
        return torch.nn.MSELoss()
    elif custom_loss == "huber":
        return HuberLoss()
    elif custom_loss == "crossentropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(
            f"Unsupported loss function: {custom_loss}. "
            f"Supported functions: mse, huber, crossentropy"
        )
