"""
Copyright (c) 2025. All rights reserved.
"""

"""
Experiment management for regression model training.

This module provides the Experiment class that orchestrates the complete
machine learning pipeline including model initialization, training context setup,
data generation, training execution, prediction, and result visualization.
"""

import datetime
import os

import torch

from lib.configs import ExperimentConfig
from dataset import prepare_data
from e_linear_reg import LinearRegressionModel
from e_non_linear_reg import MLP
from lib.loss_functions import get_loss_function
from lib.train import (
    TrainContext,
    get_lr_scheduler, 
    get_optimizer, 
    predict_model, 
    split_data, 
    train_model, 
    train_model_with_dataloader
)
from lib.utils import plot_results

class Experiment:
    """
    Complete experiment orchestrator for regression model training.
    
    This class manages the entire machine learning pipeline from model initialization
    through training, prediction, and result visualization. It provides a clean
    interface for running experiments with different configurations.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment with given configuration.
        
        Sets up the complete experiment environment including model selection,
        training context configuration, logging directory creation, and synthetic
        data generation. All components are configured based on the provided
        ExperimentConfig.
        
        Args:
            config (ExperimentConfig): Complete experiment configuration including
                                     model, training, and data parameters
        """
        self.config = config
        
        # Create logging directory for TensorBoard outputs
        self.logs_dir = "logs"
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize model and training context based on configuration
        self.model = self.define_model()
        self.train_context = self.define_train_context(self.model)

        # Generate synthetic training data
        self.inputs, self.targets = self.model.generate_data(self.config.data.fix_random_seed)
        
        # Initialize loss tracking
        self.train_loss = None
        self.val_loss = None

    
    def define_model(self) -> torch.nn.Module:
        """
        Create and configure the regression model based on experiment configuration.
        
        Factory method that instantiates either a linear regression model or
        a multi-layer perceptron based on the experiment type specified in
        the configuration.
        
        Returns:
            torch.nn.Module: Configured model instance (LinearRegressionModel or MLP)
            
        Example:
            config = ExperimentConfig(type="linear", ...)
            model = experiment.define_model()  # Returns LinearRegressionModel
        """
        if self.config.type == "linear":
            return LinearRegressionModel(self.config.model)
        else:
            return MLP(self.config.model)
        

    def define_train_context(self, model: torch.nn.Module) -> TrainContext:
        """
        Create and configure the training context with optimizers and schedulers.
        
        Sets up the complete training environment including optimizer selection,
        learning rate scheduler configuration, loss function setup, and generates
        a unique run name if not provided. All training parameters are derived
        from the experiment configuration.
        
        Args:
            model (torch.nn.Module): The model instance for which to create training context
            
        Returns:
            TrainContext: Configured training context with all necessary components
            
        Example:
            train_context = experiment.define_train_context(model)
            # Returns TrainContext with optimizer, scheduler, loss function, etc.
        """
        # Create optimizer based on configuration
        optimizer = get_optimizer(
            optimizer_type=self.config.train_config.optimizer, 
            lr=self.config.train_config.lr, 
            model=model
        )
        
        # Create learning rate scheduler
        lr_scheduler = get_lr_scheduler(
            lr=self.config.train_config.lr,
            lr_scheduler_type=self.config.train_config.lr_scheduler, 
            optimizer=optimizer, 
            epochs=self.config.train_config.epochs, 
        )
        
        # Generate unique run name if not provided
        if self.config.name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.name = f"{self.config.type}_{self.config.train_config.optimizer}_{self.config.train_config.lr_scheduler}_{self.config.train_config.epochs}_{self.config.train_config.custom_loss}_{timestamp}"
    
        return TrainContext(
            epochs=self.config.train_config.epochs, 
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_criterion=get_loss_function(self.config.train_config.custom_loss),
            log_every_k_steps=10,
            tensorboard_log_dir=self.logs_dir,
            run_name=self.config.name
        )

    def train(self) -> None:
        """
        Execute the complete training process.
        
        Orchestrates the training pipeline by choosing between DataLoader-based
        batch processing or direct tensor processing based on configuration.
        Handles data splitting, temporary file management for DataLoaders,
        and provides proper error handling with cleanup.
        
        Returns:
            None
            
        Raises:
            RuntimeError: If training fails during DataLoader processing
            
        Example:
            experiment = Experiment(config)
            experiment.train()  # Executes full training pipeline
        """ 
        # Choose training mode: DataLoader vs direct tensor processing
        if self.config.data.use_dataloader: 
            # Split data for DataLoader training
            train_inputs, train_targets, val_inputs, val_targets = split_data(self.inputs, self.targets)
            
            # Create DataLoaders and temporary CSV files
            train_dataloader, train_dataset_file_name = prepare_data(train_inputs, train_targets, suffix="_train.csv", batch_size=self.config.data.training_batch_size)
            val_dataloader, val_dataset_file_name= prepare_data(val_inputs, val_targets, suffix="_val.csv")
            
            try: 
                # Train with DataLoader (batch processing)
                self.train_loss, self.val_loss = train_model_with_dataloader(
                    self.model, 
                    self.train_context, 
                    train_dataloader, 
                    val_dataloader
                )
            except:
                
                # Clean up temporary files on error
                os.remove(train_dataset_file_name)
                os.remove(val_dataset_file_name)
                raise RuntimeError("Training Failed ")
            
            # Clean up temporary files after successful training
            os.remove(train_dataset_file_name)
            os.remove(val_dataset_file_name)

        else: 
            # Train with direct tensor processing (all data at once)
            self.train_loss, self.val_loss = train_model(
                self.model, 
                self.train_context, 
                self.inputs, 
                self.targets
            )
        
        self.save()

    def predict(self) -> torch.Tensor:
        """
        Generate predictions using the trained model on the complete dataset.
        
        Runs inference on the full dataset (both training and validation data)
        and calculates performance metrics. Logs results to TensorBoard for
        analysis and comparison across experiments.
        
        Returns:
            torch.Tensor: Model predictions as a list of numpy arrays
            
        Example:
            experiment.train()
            predictions = experiment.predict()
            # Returns predictions for visualization and analysis
        """
        return predict_model(
            self.model, 
            self.inputs, 
            self.targets, 
            self.logs_dir, 
            self.config.name
        )

    def plot_results(self, y_hat: torch.Tensor) -> None:
        """
        Create and log visualization comparing predictions vs actual targets.
        
        Generates a scatter plot showing model predictions against true target
        values and logs it to TensorBoard for visual analysis. This provides
        immediate visual feedback on model performance and learning quality.
        
        Args:
            y_hat (torch.Tensor): Model predictions from the predict() method
            
        Returns:
            None
            
        Example:
            predictions = experiment.predict()
            experiment.plot_results(predictions)
            # Creates scatter plot in TensorBoard logs
        """
        plot_results(
            self.inputs, 
            self.targets, 
            y_hat, 
            self.train_context.tensorboard_log_dir, 
            self.train_context.run_name)
        
    def save(self) -> None:
        """
        Save experiment checkpoint including model, optimizer, and loss values.
        
        Saves the complete experiment state including model weights, optimizer state,
        configuration, and final training/validation losses to a checkpoint file.
        
        Returns:
            None
        """
        checkpoint = {
            "config": self.config, 
            "model": self.model.state_dict(), 
            "optimizer": self.train_context.optimizer.state_dict(), 
            "train_loss": self.train_loss, 
            "val_loss": self.val_loss
        }
        torch.save(checkpoint, os.path.join(self.logs_dir, self.config.name + ".ckpt"))