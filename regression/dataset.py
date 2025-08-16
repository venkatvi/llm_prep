"""
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Dataset utilities for regression models.

This module provides PyTorch Dataset and DataLoader utilities for regression
tasks. Includes functionality to create datasets from CSV files and prepare
data with proper batching for training.
"""

import pandas as pd
from os import PathLike
import tempfile
import torch 
from torch.utils.data import Dataset, DataLoader 
from typing import Tuple, Union

class RegressionDataset(Dataset):
    def __init__(self, csv_file: Union[str, PathLike]) -> None:
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.inputs = torch.tensor(self.data['inputs'].values, dtype=torch.float32).view(-1, 1)
        self.targets = torch.tensor(self.data["targets"].values, dtype=torch.float32).view(-1, 1)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index:int) -> Tuple[float, float]:
        return self.inputs[index], self.targets[index]


def get_dataloader(csv_file: Union[str, PathLike], batch_size: int=32) -> None: 
    dataset = RegressionDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def generate_data_as_csv(inputs: torch.Tensor, targets: torch.Tensor, file_path: Union[str, PathLike] ) -> None: 
    data = torch.cat((inputs, targets), dim=1)
    df = pd.DataFrame(data.numpy(), columns=["inputs", "targets"])
    df.to_csv(file_path, index=False)

def prepare_data(inputs: torch.Tensor, targets: torch.Tensor, suffix: str, batch_size: int =32 ) -> Tuple[torch.utils.data.DataLoader, str]: 
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, newline="", delete=False) as csv_file: 
        csv_filename = csv_file.name
        generate_data_as_csv(inputs, targets, csv_filename)
    return get_dataloader(csv_filename, batch_size), csv_filename
        
    