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

    