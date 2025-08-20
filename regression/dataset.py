"""
Dataset utilities for regression models.
"""

import tempfile
import torch
from os import PathLike
from typing import Tuple, Union

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
        
    