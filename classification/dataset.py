"""
Copyright (c) 2025. All rights reserved.
"""

"""
CIFAR-10 dataset utilities for image classification.

This module provides utilities to load and preprocess the CIFAR-10 dataset
using PyTorch DataLoaders. It includes data transformations, normalization,
and batch processing functionality for training and evaluation.
"""

from typing import Tuple

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_cifar_dataloader() -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for CIFAR-10 training and validation datasets.

    Loads the CIFAR-10 dataset with appropriate preprocessing transformations
    including tensor conversion and normalization. Creates separate DataLoaders
    for training and validation with proper batching and shuffling.

    The preprocessing pipeline includes:
    - Converting PIL images to tensors
    - Normalizing pixel values to range [-1, 1] using mean and std of 0.5

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing:
            - train_dataloader: DataLoader for training data (50,000 images)
            - test_dataloader: DataLoader for test/validation data (10,000 images)

    Dataset Details:
        - Image size: 32x32 pixels
        - Channels: 3 (RGB)
        - Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        - Training samples: 50,000
        - Test samples: 10,000
        - Batch size: 64
        - Shuffling: Enabled for both training and validation

    Example:
        train_loader, val_loader = get_cifar_dataloader()
        for batch_images, batch_labels in train_loader:
            # batch_images: [64, 3, 32, 32]
            # batch_labels: [64] (class indices 0-9)
            break
    """
    # Define preprocessing transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL Image to tensor [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1,1]
        ]
    )

    # Load training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Load test/validation dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader
