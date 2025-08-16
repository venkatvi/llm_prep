import torch 
import matplotlib.pyplot as plt

def plot_results(inputs: torch.Tensor, targets: torch.Tensor, y_hat: torch.Tensor) -> None: 
    plt.scatter(inputs.numpy(), targets.numpy(), color="red", marker="o", label="targets")
    plt.scatter(inputs.numpy(), y_hat, color="blue", marker="*", label="y_hat" )
    plt.xlabel("inputs")
    plt.ylabel("outputs")
    plt.legend()
    plt.show()

def init_weights(layer: torch.nn.Module): 
    if isinstance(layer, torch.nn.Linear): 
        torch.nn.init.kaiming_uniform(layer.weight, nonlinearity="relu")
        if layer.bias is not None: 
            torch.nn.init.zeros_(layer.bias)
