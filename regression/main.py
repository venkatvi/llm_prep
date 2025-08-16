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
import os 
import tempfile

from e_linear_reg import LinearRegressionModel
from e_non_linear_reg import MLP

from dataset import prepare_data
from loss_functions import get_loss_function
from train import TrainContext, train, train_with_dataloader, predict, split_data, get_optimizer, get_lr_scheduler
from utils import plot_results, init_weights

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
    parser.add_argument("--type", type=str, default="linear", 
                       help="Type of regression model - linear or non-linear. Default is linear regression.")
    parser.add_argument("--epochs", type=int, default=1000, 
                       help="Number of epochs to train a model. Default is 1000.")
    parser.add_argument("--lr", type=float, default=0.01, 
                       help="learning rate for training. Default is 0.01")
    parser.add_argument("--latent_dims", type=parse_latent_dims, default=[256], 
                       help="Latent dimension for latent layer in non-linear regression models. Uses comma-separated values (e.g., '128,64,32'). Default is 256. Number of latent dims should match the number of latent layers.")
    parser.add_argument("--optimizer", type=str, default="adam", 
                       help="Type of optimizer to use. Default is Adam.")
    parser.add_argument("--lr_scheduler", type=str, default="reduceonplat", 
                       help="LR scheduler to get better performance")
    parser.add_argument("--use_dataloader", action='store_true', 
                       help="Use dataloader to iterate on data instead of large torch tensors.")
    parser.add_argument("--training_batch_size", type=int, default=8, 
                       help="Number of training samples per batch to iterate over loss computation.")
    parser.add_argument("--num_latent_layers", type=int, default=1, 
                       help="Number of latent layers to use in non linear regression model. Default is 1.")
    parser.add_argument("--custom_act", type=str, default="relu", 
                       help="Custom activation function to be enabled. Default is ReLU.")
    parser.add_argument("--allow_residual", action='store_true', 
                       help="Allow residual connections after activations in non linear reg model.")
    parser.add_argument("--custom_loss", type=str, default="mse", 
                       help="Custom Loss function to use for training loop.")
    parser.add_argument("--run_name", type=str, default=None, 
                       help="Name for this training run in TensorBoard logs.")
    args = parser.parse_args()

    # Initialize model based on type
    if args.type == "linear": 
        model = LinearRegressionModel(args.custom_act) 
    else: 
        model = MLP(args.num_latent_layers, args.latent_dims, args.custom_act, args.allow_residual)
    
    # Optional: Apply custom weight initialization
    #model.apply(init_weights)

    # Set up optimizer and learning rate scheduler
    optimizer = get_optimizer(optimizer_type=args.optimizer, lr=args.lr, model=model)
    lr_scheduler = get_lr_scheduler(lr_scheduler_type=args.lr_scheduler, optimizer=optimizer, epochs=args.epochs, lr=args.lr)
    
    # Create TensorBoard logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate unique run name if not provided
    if args.run_name is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.type}_{args.optimizer}_{timestamp}"
    
    # Create training context with all configuration
    train_context = TrainContext(
        epochs=args.epochs, 
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_criterion=get_loss_function(args.custom_loss),
        log_every_k_steps=10,
        tensorboard_log_dir=log_dir,
        run_name=args.run_name
    )

    # Generate training data
    inputs, targets = model.generate_data()

    # Choose training mode: DataLoader vs direct tensor processing
    if args.use_dataloader: 
        # Split data for DataLoader training
        train_inputs, train_targets, val_inputs, val_targets = split_data(inputs, targets)
        
        # Create DataLoaders and temporary CSV files
        train_dataloader, train_dataset_file_name = prepare_data(train_inputs, train_targets, suffix="_train.csv", batch_size=args.training_batch_size)
        val_dataloader, val_dataset_file_name= prepare_data(val_inputs, val_targets, suffix="_val.csv")
        
        try: 
            # Train with DataLoader (batch processing)
            train_with_dataloader(model, train_context, train_dataloader, val_dataloader)
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
        train(model, train_context, inputs, targets)

    # Generate predictions on full dataset
    y_hat = predict(model, inputs, targets, train_context.tensorboard_log_dir, train_context.run_name)

    # Create and log visualization plot
    plot_results(inputs, targets, y_hat, train_context.tensorboard_log_dir, train_context.run_name)
