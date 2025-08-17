import torch 
from lib.configs import ExperimentConfig, TrainConfig, DataConfig
from configs import CIFARModelConfig
from experiment import CIFARExperiment
if __name__ == "__main__":
    config = ExperimentConfig(
        type="classification", 
        name="CIFAR10_CNN", 
        train_config = TrainConfig(
            epochs=20,
            custom_loss="crossentropy",
            optimizer="adam",
            lr=0.001, 
            lr_scheduler="steplr"
        ), 
        data=DataConfig(
            use_dataloader=True,
            training_batch_size=64, 
            fix_random_seed=False, 
        ),
        model=CIFARModelConfig(
            custom_act="relu",
            num_latent_layers=3, 
            laten_dims=[], 
            allow_residual=False,
            input_channels=32
        )
    )
    experiment = CIFARExperiment(config)
    experiment.train()
    output_labels = experiment.predict()