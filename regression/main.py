"""
MIT License

Copyright (c) 2024

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
import torch 

from e_linear_reg import LinearRegressionModel
from e_non_linear_reg import MLP

from train import TrainContext, train, predict, get_optimizer, get_lr_scheduler
from utils import plot_results, init_weights


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train a regression model")
    parser.add_argument("--type", type=str, default="linear", help="Type of regression model - linear or non-linear. Default is linear regression.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train a model. Default is 1000.")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for training. Default is 0.01")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden dimension for non-linear regression models. Default is 256.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Type of optimizer to use. Default is Adam.")
    parser.add_argument("--lr_scheduler", type=str, default="reduceonplat", help="LR scheduler to get better performance")
    args = parser.parse_args()

    
    if args.type == "linear": 
        model = LinearRegressionModel() 
    else: 
        model = MLP(args.hidden_dim)
    
    #model.apply(init_weights)

    optimizer = get_optimizer(optimizer_type=args.optimizer, lr=args.lr, model=model)
    lr_scheduler = get_lr_scheduler(lr_scheduler_type=args.lr_scheduler, optimizer=optimizer, epochs=args.epochs, lr=args.lr)
    # define training context 
    train_context = TrainContext(
        epochs=args.epochs, 
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_criterion=torch.nn.MSELoss()
    )

    # generate data
    inputs, targets = model.generate_data()

    
    # Train 
    train(model, train_context, inputs, targets)

    # Predict 
    y_hat = predict(model, inputs, targets)

    # Plot two series - predictions vs targets 
    plot_results(inputs, targets, y_hat)

    