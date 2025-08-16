from dataclasses import dataclass 
import torch
from typing import Tuple

@dataclass 
class TrainContext: 
    epochs: int 
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_criterion: torch.nn.MSELoss
    log_every_k_steps: int = 10

def split_data(inputs: torch.Tensor, targets: torch.Tensor, val_split: float=0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_samples = inputs.size()[0]

    # Shuffle data
    rand_indices = torch.randperm(n_samples)
    shuffled_inputs = inputs[rand_indices]
    shuffled_targets = targets[rand_indices]

    # Split data 
    train_size=int((1-val_split)*n_samples)
    train_inputs = shuffled_inputs[:train_size]
    train_targets = shuffled_targets[:train_size]
    val_inputs = shuffled_inputs[train_size:]
    val_targets = shuffled_targets[train_size:]

    return train_inputs, train_targets, val_inputs, val_targets

def train(model: torch.nn.Module, train_context: TrainContext, inputs: torch.Tensor, targets: torch.Tensor): 
    # Split data into train and validation sets
    train_inputs, train_targets, val_inputs, val_targets = split_data(inputs, targets)
    
    for epoch in range(train_context.epochs): 
        # Training phase
        model.train()

        # forward pass 
        predictions = model(train_inputs)
        loss = train_context.loss_criterion(predictions, train_targets)

        # backward pass 
        train_context.optimizer.zero_grad()
        loss.backward()
        train_context.optimizer.step()
        

        # Step the learning rate scheduler
        if hasattr(train_context.lr_scheduler, 'step'):
            if isinstance(train_context.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                train_context.lr_scheduler.step(loss)
            else:
                train_context.lr_scheduler.step()

        # Validation phase
        if (epoch + 1) % train_context.log_every_k_steps == 0:
            model.eval()
            with torch.no_grad(): 
                val_predictions = model(val_inputs)
                val_loss = train_context.loss_criterion(val_predictions, val_targets)

            current_lr = train_context.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{train_context.epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {current_lr:.6f}")


def predict(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mse = 0.0
    y_hat = []
    with torch.no_grad():
        for x, y in zip(inputs, targets): 
            predictions = model(x)
            y_hat.append(predictions.numpy())
            mse += (predictions-y)**2
            print(f"Target: {y.item():.4f}, Actual: {predictions.item():.4f}")
        
        mse = mse/len(targets)
        print(f"MSE: {mse}")
    return y_hat 

def get_optimizer(optimizer_type: str, lr:float, model: torch.nn.Module) -> torch.optim.Optimizer: 
    if optimizer_type == "adam": 
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd": 
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == "rmsprop": 
        return torch.optim.RMSprop(model.parameters(), lr=0.001)
    else: 
        raise ValueError("Unsupported Optimizer Type")

def get_lr_scheduler(lr_scheduler_type:str, optimizer: torch.optim.Optimizer, epochs: int, lr: float) -> torch.optim.lr_scheduler.LRScheduler: 
    if lr_scheduler_type=="steplr": 
        return torch.optim.lr_scheduler.StepLR(optimizer,step_size=epochs//10, gamma=lr/100 )
    elif lr_scheduler_type == "exp": 
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr/100)
    elif lr_scheduler_type == "reduceonplat":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr/100)
    elif lr_scheduler_type == "cosine": 
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs/100, eta_min=0.001)
    else: 
        raise ValueError("Unsupported Learning Rate Scheduler Type")