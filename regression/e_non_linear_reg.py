import torch 
from typing import Tuple

class MLP(torch.nn.Module): 
    def __init__(self, hidden_dim: int): 
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_1 = torch.nn.Linear(1, self.hidden_dim)
        self.relu = torch.nn.ReLU() 
        self.linear_2 = torch.nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.linear_1(x)
        x = self.relu(x)
        return self.linear_2(x)

    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]: 
        # Define data 
        inputs = torch.rand(100, 1) * 10
        targets = 4 * inputs**2 + 2 * inputs + torch.rand(100, 1)
        return inputs, targets