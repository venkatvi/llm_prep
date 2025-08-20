"""
Loss function utilities: MSE, Huber, CrossEntropy.
"""

import torch 

class HuberLoss(torch.nn.Module):
    """Huber loss: robust regression, less sensitive to outliers."""
    
    def __init__(self, delta: float = 1):
        super().__init__()
        self.delta = delta 

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        l2_loss = 0.5 * (y_hat - y)**2
        l1_loss = torch.abs(y_hat - y)
        loss = torch.where(
            l1_loss < self.delta, 
            l2_loss, 
            self.delta * (l1_loss - 0.5 * self.delta)
        )
        return loss.mean()

def get_loss_function(custom_loss: str) -> torch.nn.Module:
    """Factory for loss functions: mse, huber, crossentropy"""
    if custom_loss == "mse":
        return torch.nn.MSELoss()
    elif custom_loss == "huber":
        return HuberLoss()
    elif custom_loss == "crossentropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {custom_loss}. "
                        f"Supported functions: mse, huber, crossentropy")