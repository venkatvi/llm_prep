"""
Copyright (c) 2025. All rights reserved.
"""

"""
Dataset utilities for regression models.
"""

import os
import sys
import tempfile
from os import PathLike
from typing import Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegressionDataset(Dataset):
    """PyTorch Dataset for regression tasks using CSV files.

    Loads regression data from CSV files with 'inputs' and 'targets' columns.
    Converts data to PyTorch tensors and provides standard Dataset interface.

    Attributes:
        data (pd.DataFrame): Raw CSV data as pandas DataFrame
        inputs (torch.Tensor): Input features as tensor [num_samples, 1]
        targets (torch.Tensor): Target values as tensor [num_samples, 1]
    """

    def __init__(self, csv_file: Union[str, PathLike]) -> None:
        """Initialize dataset from CSV file.

        Args:
            csv_file (Union[str, PathLike]): Path to CSV file with 'inputs' and 'targets' columns

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            KeyError: If CSV doesn't contain required 'inputs' and 'targets' columns
        """
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.inputs = torch.tensor(self.data["inputs"].values, dtype=torch.float32).view(-1, 1)
        self.targets = torch.tensor(self.data["targets"].values, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        """Return number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index.

        Args:
            index (int): Sample index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (input, target) tensors
        """
        return self.inputs[index], self.targets[index]


def get_dataloader(csv_file: Union[str, PathLike], batch_size: int = 32) -> DataLoader:
    """Create DataLoader from CSV file.

    Args:
        csv_file (Union[str, PathLike]): Path to CSV file containing regression data
        batch_size (int, optional): Batch size for data loading. Defaults to 32.

    Returns:
        DataLoader: PyTorch DataLoader with shuffled batches
    """
    dataset = RegressionDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def generate_data_as_csv(inputs: torch.Tensor, targets: torch.Tensor, file_path: Union[str, PathLike]) -> None:
    """Save input and target tensors as CSV file.

    Args:
        inputs (torch.Tensor): Input tensor of shape [num_samples, 1]
        targets (torch.Tensor): Target tensor of shape [num_samples, 1]
        file_path (Union[str, PathLike]): Output CSV file path
    """
    data = torch.cat((inputs, targets), dim=1)
    df = pd.DataFrame(data.numpy(), columns=["inputs", "targets"])
    df.to_csv(file_path, index=False)


def prepare_data(
    inputs: torch.Tensor, targets: torch.Tensor, suffix: str, batch_size: int = 32
) -> Tuple[DataLoader, str]:
    """Create temporary CSV file and DataLoader from tensor data.

    Converts tensor data to CSV format and creates a DataLoader for training.
    The CSV file is created as a temporary file that must be manually deleted.

    Args:
        inputs (torch.Tensor): Input tensor of shape [num_samples, 1]
        targets (torch.Tensor): Target tensor of shape [num_samples, 1]
        suffix (str): File suffix for temporary CSV file (e.g., '.csv')
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.

    Returns:
        Tuple[DataLoader, str]: Tuple containing:
            - DataLoader: Ready-to-use data loader
            - str: Path to temporary CSV file (caller must delete)
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, newline="", delete=False) as csv_file:
        csv_filename = csv_file.name
        generate_data_as_csv(inputs, targets, csv_filename)

    return get_dataloader(csv_filename, batch_size), csv_filename


def generate_polynomial_data(
    num_samples: int = 100,
    degree: int = 2,
    noise_level: float = 0.1,
    x_range: Tuple[float, float] = (0.0, 10.0),
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        targets += coeff * (inputs**i)

    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(targets) * noise_level
        targets += noise

    return inputs, targets
