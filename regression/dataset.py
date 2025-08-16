"""
Copyright (c) 2025. All rights reserved.
"""

"""
Dataset utilities for regression models.

This module provides PyTorch Dataset and DataLoader utilities for regression
tasks. Includes functionality to create datasets from CSV files and prepare
data with proper batching for training.
"""

import tempfile
import torch
from os import PathLike
from typing import Tuple, Union

import pandas as pd
from torch.utils.data import DataLoader, Dataset

class RegressionDataset(Dataset):
    """
    PyTorch Dataset for regression tasks using CSV files.
    
    This class loads regression data from CSV files containing 'inputs' and 'targets'
    columns and provides an interface for PyTorch DataLoader to iterate over
    the data in batches during training.
    """
    
    def __init__(self, csv_file: Union[str, PathLike]) -> None:
        """
        Initialize the dataset by loading data from a CSV file.
        
        Args:
            csv_file (str or PathLike): Path to CSV file containing 'inputs' and 'targets' columns
        """
        super().__init__()
        self.data = pd.read_csv(csv_file)
        
        # Convert pandas columns to PyTorch tensors
        self.inputs = torch.tensor(self.data['inputs'].values, dtype=torch.float32).view(-1, 1)
        self.targets = torch.tensor(self.data["targets"].values, dtype=torch.float32).view(-1, 1)
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of data samples
        """
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single data sample by index.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_tensor, target_tensor) pair
        """
        return self.inputs[index], self.targets[index]


def get_dataloader(csv_file: Union[str, PathLike], batch_size: int = 32) -> DataLoader:
    """
    Create a PyTorch DataLoader from a CSV file containing regression data.
    
    Args:
        csv_file (str or PathLike): Path to CSV file with 'inputs' and 'targets' columns
        batch_size (int): Number of samples per batch for training. Default is 32.
        
    Returns:
        DataLoader: PyTorch DataLoader with shuffled batches for training
    """
    dataset = RegressionDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def generate_data_as_csv(inputs: torch.Tensor, targets: torch.Tensor, file_path: Union[str, PathLike]) -> None:
    """
    Save PyTorch tensors as a CSV file for dataset creation.
    
    This function takes input and target tensors and saves them as a CSV file
    with 'inputs' and 'targets' columns that can be loaded by RegressionDataset.
    
    Args:
        inputs (torch.Tensor): Input feature tensors
        targets (torch.Tensor): Target value tensors
        file_path (str or PathLike): Path where CSV file should be saved
        
    Returns:
        None
    """
    # Concatenate inputs and targets along feature dimension
    data = torch.cat((inputs, targets), dim=1)
    
    # Convert to pandas DataFrame and save as CSV
    df = pd.DataFrame(data.numpy(), columns=["inputs", "targets"])
    df.to_csv(file_path, index=False)

def prepare_data(inputs: torch.Tensor, targets: torch.Tensor, suffix: str, batch_size: int = 32) -> Tuple[DataLoader, str]:
    """
    Create a temporary CSV file and DataLoader from tensor data.
    
    This convenience function creates a temporary CSV file from tensor data,
    then returns a DataLoader for batch processing. The temporary file should
    be cleaned up by the caller after use.
    
    Args:
        inputs (torch.Tensor): Input feature tensors
        targets (torch.Tensor): Target value tensors  
        suffix (str): File suffix for the temporary CSV file (e.g., "_train.csv")
        batch_size (int): Number of samples per batch. Default is 32.
        
    Returns:
        Tuple[DataLoader, str]: DataLoader for batched iteration and temporary file path
        
    Note:
        The caller is responsible for deleting the temporary CSV file after use.
        
    Example:
        dataloader, temp_file = prepare_data(x, y, "_train.csv", batch_size=64)
        # ... use dataloader for training ...
        os.remove(temp_file)  # Clean up temporary file
    """
    # Create temporary CSV file (delete=False means caller must clean up)
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, newline="", delete=False) as csv_file: 
        csv_filename = csv_file.name
        generate_data_as_csv(inputs, targets, csv_filename)
    
    return get_dataloader(csv_filename, batch_size), csv_filename
        
    