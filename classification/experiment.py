"""
Copyright (c) 2025. All rights reserved.
"""

"""
CIFAR-10 classification experiment management.

This module provides the CIFARExperiment class that orchestrates the complete
machine learning pipeline for image classification on the CIFAR-10 dataset,
including model training, validation, prediction, and accuracy evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
from lib.configs import ExperimentConfig
from cifar_cnn import CIFARCNN
from lib.experiment import Experiment
from lib.train import train_model_with_dataloader
from dataset import get_cifar_dataloader
from lib.logger import Logger

class CIFARExperiment(Experiment):
    """
    CIFAR-10 classification experiment orchestrator.
    
    This class manages the complete machine learning pipeline for CIFAR-10 image
    classification, including CNN model training, validation, prediction, and
    accuracy evaluation. It extends the base Experiment class to handle
    image classification specific workflows.
    
    Attributes:
        model (CIFARCNN): Convolutional neural network model for image classification
        logger (Logger): TensorBoard logger for metrics and visualization tracking
        accuracy (float): Classification accuracy on validation set
    """
    
    def __init__(self, config: ExperimentConfig) -> None:
        """
        Initialize the CIFAR-10 classification experiment.
        
        Sets up the experiment environment including model instantiation,
        training context configuration, and logging infrastructure specifically
        for image classification tasks.
        
        Args:
            config (ExperimentConfig): Complete experiment configuration including
                                     model architecture, training parameters, and data settings
        """
        super().__init__(config)
        self.model = self.define_model()
        self.logger = Logger(self.logs_dir, self.config.name)
    
    def define_model(self) -> torch.nn.Module:
        """
        Create and configure the CIFAR-10 CNN model.
        
        Instantiates a convolutional neural network specifically designed for
        CIFAR-10 image classification with the configured number of input channels.
        
        Returns:
            torch.nn.Module: CIFARCNN model instance configured for the experiment
            
        Example:
            model = experiment.define_model()  # Returns CIFARCNN with 3 input channels
        """
        return CIFARCNN(self.config.model.input_channels)

    def train(self) -> None:
        """
        Execute the complete CIFAR-10 training process.
        
        Orchestrates the training pipeline by loading CIFAR-10 data through DataLoaders,
        then executing the training loop with automatic batch processing, loss computation,
        and validation evaluation. Updates training and validation loss values.
        
        The training process includes:
        - Loading CIFAR-10 train/validation DataLoaders
        - Batch-wise forward and backward propagation
        - Optimizer step and learning rate scheduling
        - Validation evaluation and loss tracking
        - TensorBoard logging of metrics
        
        Returns:
            None
            
        Side Effects:
            Updates self.train_loss and self.val_loss with final epoch values
        """
        train_dataloader, val_dataloader = get_cifar_dataloader()
        self.train_loss, self.val_loss = train_model_with_dataloader(
            self.model, 
            self.train_context, 
            train_dataloader, 
            val_dataloader
        )
    
    def predict(self) -> torch.Tensor:
        """
        Generate predictions and evaluate classification accuracy.
        
        Runs the trained model on the CIFAR-10 validation dataset to generate
        class predictions and compute overall classification accuracy. The model
        outputs class probabilities which are converted to predicted class labels
        using argmax operation.
        
        Returns:
            torch.Tensor: Concatenated tensor of predicted class labels for all validation samples
            
        Side Effects:
            - Updates self.accuracy with validation accuracy
            - Logs accuracy metric to TensorBoard
            
        Example:
            predictions = experiment.predict()
            print(f"Model accuracy: {experiment.accuracy:.4f}")
            print(f"Predictions shape: {predictions.shape}")  # [num_validation_samples]
        """
        _, val_dataloader = get_cifar_dataloader()
        accuracy = 0.0
        num_samples = 0
        all_predictions = []
        
        # Evaluate model on validation set
        for batch_inputs, batch_targets in val_dataloader: 
            predictions = self.model(batch_inputs)
            output_labels = predictions.argmax(dim=1)
            accuracy += sum(output_labels == batch_targets)
            num_samples += len(batch_targets)
            all_predictions.append(output_labels)
            
        # Calculate and log accuracy
        self.accuracy = accuracy/num_samples
        self.logger.log_scalars({"accuracy": self.accuracy})
        
        return torch.cat(all_predictions)
    
    def plot_results(self, y_hat: torch.Tensor) -> None:
        """
        Create visualizations for classification results.
        
        Placeholder method for generating classification-specific visualizations
        such as confusion matrices, class distribution plots, or sample predictions
        with confidence scores. Currently not implemented.
        
        Args:
            y_hat (torch.Tensor): Predicted class labels from the model
            
        Returns:
            None
            
        Note:
            Future implementations could include:
            - Confusion matrix visualization
            - Per-class accuracy breakdown
            - Sample images with predictions
            - Confidence score distributions
        """
        pass
    

