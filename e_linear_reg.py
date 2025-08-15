""" Linear regression model """
from dataclasses import dataclass 
import torch 
import matplotlib as plt
@dataclass 
class TrainContext: 
    epochs: int 
    optimizer: torch.optim.Optimizer
    loss_criterion: torch.nn.MSELoss
    log_every_k_steps: int = 10

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x:torch.tensor)->torch.tensor: 
        return self.linear(x)


def train(model: torch.nn.Module, train_context: TrainContext, inputs: torch.tensor, targets: torch.tensor): 
    for epoch in range(train_context.epochs): 
        # forward pass 
        predictions = model(inputs)
        loss = train_context.loss_criterion(predictions, targets)

        # backward pass 
        train_context.optimizer.zero_grad()
        loss.backward()
        train_context.optimizer.step()

        # Log progress every k steps
        if (epoch + 1) % train_context.log_every_k_steps == 0: 
            print(f"Epoch {epoch+1}/{train_context.epochs}, Loss: {loss.item():.4f}")


if __name__ == "__main__": 
    # define model 
    model = LinearRegressionModel() 

    # define training context 
    train_context = TrainContext(
        epochs=1000, 
        optimizer=torch.optim.SGD(model.parameters()),
        loss_criterion=torch.nn.MSELoss()
    )

    # Define data 
    inputs = torch.rand(100, 1) * 10
    targets = 4 * inputs**2 + 3 * inputs + torch.rand(100, 1)

    print(type(inputs))

    # Train 
    train(model, train_context, inputs, targets)

    # Predict 
    mse = 0.0
    for x, y in zip(inputs, targets): 
        predictions = model(x)
        mse += (predictions-y)**2
        print(f"Target: {y.item():.4f}, Actual: {predictions.item():.4f}")
    
    mse = mse/len(targets)
    print(f"MSE: {mse}")

    # Plot two series - predictions vs targets 
    
    