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
from abc import ABC, abstractmethod
from typing import Optional

import torch

from lib.configs import ExperimentConfig
from lib.loss_functions import get_loss_function
from lib.train import (
    TrainContext,
    get_lr_scheduler,
    get_optimizer,
)


class Experiment(ABC):
    """
    Complete experiment orchestrator for regression model training.

    This class manages the entire machine learning pipeline from model initialization
    through training, prediction, and result visualization. It provides a clean
    interface for running experiments with different configurations.
    """

    def __init__(self, config: ExperimentConfig) -> None:
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
        self.train_loss: Optional[float] = None
        self.val_loss: Optional[float] = None

    @abstractmethod
    def define_model(self) -> torch.nn.Module:
        pass

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
            model=model,
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
            self.config.name = (
                f"{self.config.type}_{self.config.train_config.optimizer}_"
                f"{self.config.train_config.lr_scheduler}_{self.config.train_config.epochs}_"
                f"{self.config.train_config.custom_loss}_{timestamp}"
            )

        return TrainContext(
            epochs=self.config.train_config.epochs,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_criterion=get_loss_function(self.config.train_config.custom_loss),
            log_every_k_steps=10,
            tensorboard_log_dir=self.logs_dir,
            run_name=self.config.name,
        )

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self) -> torch.Tensor:
        pass

    @abstractmethod
    def plot_results(self, y_hat: torch.Tensor) -> None:
        pass

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
            "val_loss": self.val_loss,
        }
        torch.save(checkpoint, os.path.join(self.logs_dir, self.config.name + ".ckpt"))
