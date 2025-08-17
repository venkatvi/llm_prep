import os
import torch 
from dataset import prepare_data
from e_linear_reg import LinearRegressionModel
from e_non_linear_reg import MLP
from lib.experiment import Experiment
from lib.configs import ExperimentConfig
from lib.train import train_model, predict_model, train_model_with_dataloader
from lib.utils import plot_results
class RegressionExperiment(Experiment): 
    def __init__(self, config: ExperimentConfig) -> None: 
        super().__init__(config)
        self.model = self.define_model()
    
    def define_model(self) -> torch.nn.Module:
        """
        Create and configure the regression model based on experiment configuration.
        
        Factory method that instantiates either a linear regression model or
        a multi-layer perceptron based on the experiment type specified in
        the configuration.
        
        Returns:
            torch.nn.Module: Configured model instance (LinearRegressionModel or MLP)
            
        Example:
            config = ExperimentConfig(type="linear", ...)
            model = experiment.define_model()  # Returns LinearRegressionModel
        """
        if self.config.type == "linear":
            return LinearRegressionModel(self.config.model)
        else:
            return MLP(self.config.model)
    
    def train(self) -> None:
        """
        Execute the complete training process.
        
        Orchestrates the training pipeline by choosing between DataLoader-based
        batch processing or direct tensor processing based on configuration.
        Handles data splitting, temporary file management for DataLoaders,
        and provides proper error handling with cleanup.
        
        Returns:
            None
            
        Raises:
            RuntimeError: If training fails during DataLoader processing
            
        Example:
            experiment = Experiment(config)
            experiment.train()  # Executes full training pipeline
        """ 
        # Choose training mode: DataLoader vs direct tensor processing
        if self.config.data.use_dataloader: 
            # Split data for DataLoader training
            train_inputs, train_targets, val_inputs, val_targets = split_data(self.inputs, self.targets)
            
            # Create DataLoaders and temporary CSV files
            train_dataloader, train_dataset_file_name = prepare_data(train_inputs, train_targets, suffix="_train.csv", batch_size=self.config.data.training_batch_size)
            val_dataloader, val_dataset_file_name= prepare_data(val_inputs, val_targets, suffix="_val.csv")
            
            try: 
                # Train with DataLoader (batch processing)
                self.train_loss, self.val_loss = train_model_with_dataloader(
                    self.model, 
                    self.train_context, 
                    train_dataloader, 
                    val_dataloader
                )
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
            self.train_loss, self.val_loss = train_model(
                self.model, 
                self.train_context, 
                self.inputs, 
                self.targets
            )
        
        self.save()

    def predict(self) -> torch.Tensor:
        """
        Generate predictions using the trained model on the complete dataset.
        
        Runs inference on the full dataset (both training and validation data)
        and calculates performance metrics. Logs results to TensorBoard for
        analysis and comparison across experiments.
        
        Returns:
            torch.Tensor: Model predictions as a list of numpy arrays
            
        Example:
            experiment.train()
            predictions = experiment.predict()
            # Returns predictions for visualization and analysis
        """
        return predict_model(
            self.model, 
            self.inputs, 
            self.targets, 
            self.logs_dir, 
            self.config.name
        )
    def plot_results(self, y_hat: torch.Tensor) -> None:
        """
        Create and log visualization comparing predictions vs actual targets.
        
        Generates a scatter plot showing model predictions against true target
        values and logs it to TensorBoard for visual analysis. This provides
        immediate visual feedback on model performance and learning quality.
        
        Args:
            y_hat (torch.Tensor): Model predictions from the predict() method
            
        Returns:
            None
            
        Example:
            predictions = experiment.predict()
            experiment.plot_results(predictions)
            # Creates scatter plot in TensorBoard logs
        """
        plot_results(
            self.inputs, 
            self.targets, 
            y_hat, 
            self.train_context.tensorboard_log_dir, 
            self.train_context.run_name)