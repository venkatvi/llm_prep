import torch 
from lib.configs import ExperimentConfig
from cifar_cnn import CIFARCNN
from lib.experiment import Experiment
from lib.train import train_model_with_dataloader
from dataset import get_cifar_dataloader
from lib.logger import Logger
class CIFARExperiment(Experiment): 
    def __init__(self, config:ExperimentConfig):
        super().__init__()
        self.model = self.define_model()
        self.logger = Logger(self.logs_dir, self.config.name)
    
    def define_model(self) -> torch.nn.Module: 
        return CIFARCNN(self.config.model.input_channels)

    def train(self) -> None: 
        train_dataloader, val_dataloader = get_cifar_dataloader()
        self.train_loss, self.val_loss = train_model_with_dataloader(
            self.model, 
            self.train_context, 
            train_dataloader, 
            val_dataloader
        )
    
    def predict(self) -> torch.Tensor: 
        _, val_dataloader = get_cifar_dataloader()
        accuracy = 0.0
        num_samples = 0
        all_predictions = []
        for batch_inputs, batch_targets in val_dataloader: 
            predictions = self.model(batch_inputs)
            output_labels = predictions.argmax(dim=1)
            accuracy += sum(output_labels == batch_targets)
            num_samples += len(batch_targets)
            all_predictions.append(output_labels)
        self.accuracy = accuracy/num_samples
        self.logger.log_scalars({"accuracy": self.accuracy})
        return torch.cat(all_predictions)
    
    def plot_results(self, y_hat: torch.Tensor) -> None:
        """
        TBD
        """
        pass
    

