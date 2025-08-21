"""
Copyright (c) 2025. All rights reserved.
"""

"""
Configuration dataclasses for regression experiments.

This module provides structured configuration classes using Python dataclasses
to organize and validate experiment parameters. Each config class groups related
parameters for different aspects of the machine learning pipeline.
"""

from dataclasses import dataclass


@dataclass 
class TrainConfig:
    """
    Configuration for training loop parameters.
    
    Groups all training-related hyperparameters including epochs, loss functions,
    optimizers, learning rates, and scheduling strategies. This configuration
    is used to set up the training context and optimize model convergence.
    
    Attributes:
        epochs (int): Number of training epochs to run
        custom_loss (str): Loss function type ("mse", "huber", "crossentropy")
        optimizer (str): Optimizer algorithm ("adam", "sgd", "rmsprop")
        lr (float): Learning rate for gradient descent optimization
        lr_scheduler (str): Learning rate scheduler ("steplr", "exp", "reduceonplat", "cosine")
        step_size (int): Step size parameter for learning rate schedulers (StepLR, CosineAnnealing)
        
    Example:
        train_config = TrainConfig(
            epochs=1000,
            custom_loss="mse",
            optimizer="adam",
            lr=0.001,
            lr_scheduler="steplr",
            step_size=10
        )
    """
    epochs: int                 # Number of training epochs
    custom_loss: str           # Loss function type 
    optimizer: str             # Optimizer algorithm
    lr: float                  # Learning rate
    lr_scheduler: str          # Learning rate scheduler
    step_size: int             # Step size for learning rate schedulers


@dataclass 
class DataConfig:
    """
    Configuration for data processing and loading.
    
    Defines how training data is processed, loaded, and managed during training.
    Controls batch processing strategies and reproducibility settings for
    consistent experiment results across runs.
    
    Attributes:
        use_dataloader (bool): Whether to use PyTorch DataLoader for batch processing
        training_batch_size (int): Number of samples per training batch
        fix_random_seed (bool): Whether to fix random seeds for reproducibility
        
    Example:
        data_config = DataConfig(
            use_dataloader=True,
            training_batch_size=32,
            fix_random_seed=True
        )
    """
    use_dataloader: bool       # Enable DataLoader batch processing
    training_batch_size: int   # Batch size for training
    fix_random_seed: bool      # Fix random seeds for reproducibility


@dataclass 
class ModelConfig: 
    name: str

@dataclass 
class ExperimentConfig:
    """
    Complete experiment configuration combining all parameter groups.
    
    Top-level configuration class that aggregates all experiment parameters
    including model architecture, training settings, and data processing
    options. Provides a single configuration object for running experiments.
    
    Attributes:
        type (str): Experiment type ("linear" or "nlinear")
        name (str): Unique name for this experiment run
        train_config (TrainConfig): Training loop configuration
        data (DataConfig): Data processing configuration  
        model (ModelConfig): Model architecture configuration
        
    Example:
        config = ExperimentConfig(
            type="nlinear",
            name="mlp_experiment_001",
            train_config=TrainConfig(...),
            data=DataConfig(...),
            model=ModelConfig(...)
        )
    """
    type: str                  # Experiment type (linear/nlinear)
    name: str                  # Experiment run name
    train_config: TrainConfig  # Training configuration
    data: DataConfig           # Data processing configuration
    model: ModelConfig         # Model architecture configuration