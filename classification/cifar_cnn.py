"""
Copyright (c) 2025. All rights reserved.
"""

"""
Convolutional Neural Network for CIFAR-10 image classification.

This module implements a CNN architecture specifically designed for CIFAR-10
dataset classification. The network uses convolutional layers with ReLU activation,
max pooling for spatial dimension reduction, and fully connected layers for
final classification.
"""

import torch
import torch.nn.functional as F


class CIFARCNN(torch.nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 image classification.

    A CNN architecture specifically designed for CIFAR-10 dataset (32x32 RGB images)
    with 10 output classes. The network consists of two convolutional blocks followed
    by fully connected layers for classification.

    Architecture:
        - Conv Block 1: [Conv2d(3->32) + ReLU + Conv2d(32->32) + ReLU + MaxPool2d] -> 16x16
        - Conv Block 2: [Conv2d(32->64) + ReLU + Conv2d(64->64) + ReLU + MaxPool2d] -> 8x8
        - Classifier: [Flatten + Linear(4096->512) + Linear(512->10)]

    Input Shape: [batch_size, input_channels, 32, 32]
    Output Shape: [batch_size, 10] (class logits)

    Attributes:
        conv_1 (nn.Conv2d): First convolutional layer (input_channels -> 32)
        conv_2 (nn.Conv2d): Second convolutional layer (32 -> 32)
        maxpool_1 (nn.MaxPool2d): First max pooling layer (reduces to 16x16)
        conv_3 (nn.Conv2d): Third convolutional layer (32 -> 64)
        conv_4 (nn.Conv2d): Fourth convolutional layer (64 -> 64)
        maxpool_2 (nn.MaxPool2d): Second max pooling layer (reduces to 8x8)
        flatten (nn.Flatten): Flattens feature maps for fully connected layers
        fc_1 (nn.Linear): First fully connected layer (4096 -> 512)
        fc_2 (nn.Linear): Output layer (512 -> 10 classes)
    """

    def __init__(self, input_channels: int) -> None:
        """
        Initialize the CIFAR CNN architecture.

        Args:
            input_channels (int): Number of input color channels (typically 3 for RGB)

        Example:
            model = CIFARCNN(input_channels=3)  # For RGB CIFAR-10 images
        """
        super().__init__()

        # First convolutional block: input -> 32 feature maps
        self.conv_1 = torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16

        # Second convolutional block: 32 -> 64 feature maps
        self.conv_3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8

        # Classifier layers
        self.flatten = torch.nn.Flatten()
        self.fc_1 = torch.nn.Linear(64 * 8 * 8, 512)  # 4096 -> 512
        self.fc_2 = torch.nn.Linear(512, 10)  # 512 -> 10 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Processes input images through convolutional blocks with ReLU activations
        and max pooling, followed by fully connected layers for classification.

        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, channels, 32, 32]

        Returns:
            torch.Tensor: Class logits of shape [batch_size, 10]

        Example:
            model = CIFARCNN(3)
            input_batch = torch.randn(32, 3, 32, 32)  # Batch of 32 CIFAR images
            logits = model(input_batch)  # Shape: [32, 10]
            predictions = torch.softmax(logits, dim=1)  # Convert to probabilities
        """
        # First convolutional block
        x = F.relu(self.conv_1(x))  # [batch, 32, 32, 32]
        x = F.relu(self.conv_2(x))  # [batch, 32, 32, 32]
        x = self.maxpool_1(x)  # [batch, 32, 16, 16]

        # Second convolutional block
        x = F.relu(self.conv_3(x))  # [batch, 64, 16, 16]
        x = F.relu(self.conv_4(x))  # [batch, 64, 16, 16]
        x = self.maxpool_2(x)  # [batch, 64, 8, 8]

        # Classifier
        x = self.flatten(x)  # [batch, 4096]
        x = F.relu(self.fc_1(x))  # [batch, 512]
        return self.fc_2(x)  # [batch, 10]


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the CIFARCNN model.

    Demonstrates model instantiation, forward pass, and output shape verification
    for CIFAR-10 classification tasks.
    """
    # Create model for CIFAR-10 (3 RGB channels)
    model = CIFARCNN(input_channels=3)

    # Create test input: batch_size=1, channels=3, height=32, width=32
    test_input = torch.randn(1, 3, 32, 32)

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(test_input)

    print(f"Model output shape: {output.shape}")  # Expected: [1, 10]
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify output can be converted to class probabilities
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Class probabilities: {probabilities.squeeze()}")
