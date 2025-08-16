"""
Copyright (c) 2025. All rights reserved.
"""

"""
Loss function utilities for regression models.

This module provides custom loss functions and a factory function to create
various PyTorch loss functions. Includes MSE, Huber loss, and CrossEntropy
for different training scenarios.
"""

import torch 

class HuberLoss(torch.nn.Module):
    """
    Huber Loss function for robust regression.
    
    Huber loss is less sensitive to outliers than squared error loss.
    It uses squared error when the absolute error is small (< delta) and
    linear error when the absolute error is large (>= delta).
    
    This makes it more robust to outliers compared to Mean Squared Error
    while still being differentiable everywhere.
    """
    
    def __init__(self, delta: float = 1):
        """
        Initialize Huber Loss with specified delta threshold.
        
        Args:
            delta (float): Threshold for switching between L2 and L1 loss.
                          Controls sensitivity to outliers. Default is 1.0.
        """
        super().__init__()
        self.delta = delta 

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss between predictions and targets.
        
        Args:
            y (torch.Tensor): True target values
            y_hat (torch.Tensor): Predicted values from model
            
        Returns:
            torch.Tensor: Scalar loss value (mean across all elements)
        """
        # Calculate L2 (squared) and L1 (absolute) components
        l2_loss = 0.5 * (y_hat - y)**2
        l1_loss = torch.abs(y_hat - y)
        
        # Use L2 loss for small errors, L1 loss for large errors 
        loss = torch.where(
            l1_loss < self.delta, 
            l2_loss, 
            self.delta * (l1_loss - 0.5 * self.delta)
        )
        return loss.mean()  # Return single scalar value 

def get_loss_function(custom_loss: str) -> torch.nn.Module:
    """
    Factory function to create PyTorch loss function instances.
    
    This function returns the appropriate PyTorch loss function based on the
    provided string identifier. Supports common loss functions for regression
    and classification tasks.
    
    Args:
        custom_loss (str): Name of the loss function to create.
                          Supported values:
                          - "mse": Mean Squared Error (good for regression)
                          - "huber": Huber Loss (robust regression, less sensitive to outliers)
                          - "crossentropy": Cross Entropy Loss (for classification)
    
    Returns:
        torch.nn.Module: Initialized PyTorch loss function
    
    Raises:
        ValueError: If custom_loss is not a supported loss function
        
    Example:
        loss_fn = get_loss_function("mse")     # Returns nn.MSELoss()
        loss_fn = get_loss_function("huber")   # Returns HuberLoss()
    """
    if custom_loss == "mse":
        return torch.nn.MSELoss()
    elif custom_loss == "huber":
        return HuberLoss()
    elif custom_loss == "crossentropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {custom_loss}. "
                        f"Supported functions: mse, huber, crossentropy")