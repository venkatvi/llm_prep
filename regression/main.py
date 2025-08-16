"""
Copyright (c) 2025. All rights reserved.
"""

"""
Main entry point for regression model training.

This module provides a command-line interface for training linear and non-linear
regression models using PyTorch. It supports various optimizers, learning rate
schedulers, and model configurations.
"""

import argparse

from configs import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from experiment import Experiment

def parse_latent_dims(value):
    """
    Parse comma-separated string of integers into list.
    
    Args:
        value (str): Comma-separated string like "128,64,32"
        
    Returns:
        list[int]: List of integer dimensions
    """
    return [int(x.strip()) for x in value.split(',')]


if __name__ == "__main__": 
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Train a regression model")
    # Experiment name 
    parser.add_argument("--type", type=str, default="linear", 
                       help="Type of regression model - linear or non-linear. Default is linear regression.")
    parser.add_argument("--run_name", type=str, default=None, 
                       help="Name for this training run in TensorBoard logs.")
    # Train Loop 
    parser.add_argument("--epochs", type=int, default=1000, 
                       help="Number of epochs to train a model. Default is 1000.")
    parser.add_argument("--custom_loss", type=str, default="mse", 
                       help="Custom Loss function to use for training loop.")
    parser.add_argument("--optimizer", type=str, default="adam", 
                       help="Type of optimizer to use. Default is Adam.")
    ## Optimizer and Learning Rate 
    parser.add_argument("--lr", type=float, default=0.01, 
                       help="learning rate for training. Default is 0.01")
    parser.add_argument("--lr_scheduler", type=str, default="reduceonplat", 
                       help="LR scheduler to get better performance")
    # Data 
    parser.add_argument("--use_dataloader", action='store_true', 
                       help="Use dataloader to iterate on data instead of large torch tensors.")
    parser.add_argument("--training_batch_size", type=int, default=8, 
                       help="Number of training samples per batch to iterate over loss computation.")
    parser.add_argument("--fix_random_seed", action='store_true', help="Fix random data gen seed for finding consistencies between runs.")
    # Model 
    parser.add_argument("--custom_act", type=str, default="relu", 
                       help="Custom activation function to be enabled. Default is ReLU.")
    parser.add_argument("--num_latent_layers", type=int, default=1, 
                       help="Number of latent layers to use in non linear regression model. Default is 1.")
    parser.add_argument("--latent_dims", type=parse_latent_dims, default=[256], 
                       help="Latent dimension for latent layer in non-linear regression models. Uses comma-separated values (e.g., '128,64,32'). Default is 256. Number of latent dims should match the number of latent layers.")
    parser.add_argument("--allow_residual", action='store_true', 
                       help="Allow residual connections after activations in non linear reg model.")
    args = parser.parse_args()

    experiment_config = ExperimentConfig(
        type=args.type, 
        name=args.run_name,
        train_config=TrainConfig(
            epochs=args.epochs, 
            custom_loss=args.custom_loss, 
            optimizer=args.optimizer, 
            lr_scheduler=args.lr_scheduler,
            lr = args.lr
        ), 
        data=DataConfig(
            use_dataloader=args.use_dataloader, 
            training_batch_size=args.training_batch_size, 
            fix_random_seed=args.fix_random_seed
        ),
        model=ModelConfig(
            custom_act=args.custom_act, 
            num_latent_layers=args.num_latent_layers,
            latent_dims=args.latent_dims, 
            allow_residual=args.allow_residual
        )
    )

    experiment = Experiment(experiment_config)
    
    # Train 
    experiment.train()

    # Generate predictions on full dataset
    y_hat = experiment.predict()

    # Create and log visualization plot
    experiment.plot_results(y_hat)
    