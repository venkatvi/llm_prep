"""
Dataset utilities for regression models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import torch
from os import PathLike
from typing import Tuple, Union, Optional

import pandas as pd
from torch.utils.data import DataLoader, Dataset

class RegressionDataset(Dataset):
    """PyTorch Dataset for regression tasks using CSV files."""
    
    def __init__(self, csv_file: Union[str, PathLike]) -> None:
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.inputs = torch.tensor(self.data['inputs'].values, dtype=torch.float32).view(-1, 1)
        self.targets = torch.tensor(self.data["targets"].values, dtype=torch.float32).view(-1, 1)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def get_dataloader(csv_file: Union[str, PathLike], batch_size: int = 32) -> DataLoader:
    """Create DataLoader from CSV file."""
    dataset = RegressionDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def generate_data_as_csv(inputs: torch.Tensor, targets: torch.Tensor, file_path: Union[str, PathLike]) -> None:
    """Save tensors as CSV file."""
    data = torch.cat((inputs, targets), dim=1)
    df = pd.DataFrame(data.numpy(), columns=["inputs", "targets"])
    df.to_csv(file_path, index=False)

def prepare_data(inputs: torch.Tensor, targets: torch.Tensor, suffix: str, batch_size: int = 32) -> Tuple[DataLoader, str]:
    """Create temporary CSV file and DataLoader from tensor data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, newline="", delete=False) as csv_file: 
        csv_filename = csv_file.name
        generate_data_as_csv(inputs, targets, csv_filename)
    
    return get_dataloader(csv_filename, batch_size), csv_filename

def generate_polynomial_data(num_samples: int = 100, degree: int = 2, noise_level: float = 0.1, x_range: Tuple[float, float] = (0.0, 10.0), random_seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic polynomial regression data.
    
    Args:
        num_samples: Number of data points to generate
        degree: Polynomial degree (1=linear, 2=quadratic, etc.)
        noise_level: Standard deviation of Gaussian noise
        x_range: Range for input values (min, max)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (inputs, targets) tensors
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Generate input values
    x_min, x_max = x_range
    inputs = torch.rand(num_samples, 1) * (x_max - x_min) + x_min
    
    # Generate polynomial coefficients
    coefficients = torch.randn(degree + 1) * 2.0  # Random coefficients
    
    # Compute polynomial targets: y = c0 + c1*x + c2*x^2 + ... + cn*x^n
    targets = torch.zeros(num_samples, 1)
    for i, coeff in enumerate(coefficients):
        targets += coeff * (inputs ** i)
    
    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(targets) * noise_level
        targets += noise
    
    return inputs, targets