""" Linear regression model """
import torch 
from typing import Tuple 
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x:torch.Tensor)->torch.Tensor: 
        return self.linear(x)

    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]: 
        # Define data 
        inputs = torch.rand(100, 1) * 10
        targets = 100* inputs + torch.rand(100, 1)
        return inputs, targets