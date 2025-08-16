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
Main entry point for regression model training.

This module provides a command-line interface for training linear and non-linear
regression models using PyTorch. It supports various optimizers, learning rate
schedulers, and model configurations.
"""

import argparse 
import tempfile
import torch 
import os 

from e_linear_reg import LinearRegressionModel
from e_non_linear_reg import MLP

from dataset import prepare_data
from loss_functions import get_loss_function
from train import TrainContext, train, train_with_dataloader, predict, split_data, get_optimizer, get_lr_scheduler
from utils import plot_results, init_weights


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train a regression model")
    parser.add_argument("--type", type=str, default="linear", help="Type of regression model - linear or non-linear. Default is linear regression.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train a model. Default is 1000.")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for training. Default is 0.01")
    def parse_latent_dims(value):
        return [int(x.strip()) for x in value.split(',')]
    
    parser.add_argument("--latent_dims", type=parse_latent_dims, default=[256], help="Latent dimension for latent layer in non-linear regression models. Uses comma-separated values (e.g., '128,64,32'). Default is 256. Number of latent dims should match the number of latent layers.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Type of optimizer to use. Default is Adam.")
    parser.add_argument("--lr_scheduler", type=str, default="reduceonplat", help="LR scheduler to get better performance")
    parser.add_argument("--use_dataloader", action='store_true', help="Use dataloader to iterate on data instead of large torch tensors.")
    parser.add_argument("--training_batch_size", type=int, default=8, help="Number of training samples per batch to iterate over loss computation.")
    parser.add_argument("--num_latent_layers", type=int, default=1, help="Number of latent layers to use in non linear regression model. Default is 1.")
    parser.add_argument("--custom_act", type=str, default="relu", help="Custom activation function to be enabled. Default is ReLU.")
    parser.add_argument("--allow_residual", action='store_true', help="Allow residual connections after activations in non linear reg model.")
    parser.add_argument("--custom_loss", type=str, default="mse", help="Custom Loss function to use for training loop.")
    args = parser.parse_args()

    
    if args.type == "linear": 
        model = LinearRegressionModel(args.custom_act) 
    else: 
        model = MLP(args.num_latent_layers, args.latent_dims, args.custom_act, args.allow_residual)
    
    #model.apply(init_weights)

    optimizer = get_optimizer(optimizer_type=args.optimizer, lr=args.lr, model=model)
    lr_scheduler = get_lr_scheduler(lr_scheduler_type=args.lr_scheduler, optimizer=optimizer, epochs=args.epochs, lr=args.lr)
    
    # define training context 
    train_context = TrainContext(
        epochs=args.epochs, 
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_criterion=get_loss_function(args.custom_loss)
    )

    # generate data
    inputs, targets = model.generate_data()

    # With dataset usage 
    if args.use_dataloader: 
        train_inputs, train_targets, val_inputs, val_targets = split_data(inputs, targets)
        
        train_dataloader, train_dataset_file_name = prepare_data(train_inputs, train_targets, suffix="_train.csv", batch_size=args.training_batch_size)
        val_dataloader, val_dataset_file_name= prepare_data(val_inputs, val_targets, suffix="_val.csv")
        
        try: 
            train_with_dataloader(model, train_context, train_dataloader, val_dataloader)
        except: 
            os.remove(train_dataset_file_name)
            os.remove(val_dataset_file_name)
            raise RuntimeError("Training Failed ")
        
        os.remove(train_dataset_file_name)
        os.remove(val_dataset_file_name)

    else: 
        # Train 
        train(model, train_context, inputs, targets)

    # Predict 
    y_hat = predict(model, inputs, targets)

    # Plot two series - predictions vs targets 
    plot_results(inputs, targets, y_hat)

    