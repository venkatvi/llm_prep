"""
Copyright (c) 2025. All rights reserved.
"""

"""
Main entry point for CIFAR-10 image classification training.

This module provides the main execution script for training convolutional neural
networks on the CIFAR-10 dataset using PyTorch. It demonstrates the configuration
and execution of classification experiments with the experiment framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
from lib.configs import ExperimentConfig, TrainConfig, DataConfig
from configs import CIFARModelConfig
from experiment import CIFARExperiment
if __name__ == "__main__":
    config = ExperimentConfig(
        type="classification", 
        name="CIFAR10_CNN", 
        train_config = TrainConfig(
            epochs=5,
            custom_loss="crossentropy",
            optimizer="adam",
            lr=0.001, 
            lr_scheduler="steplr",
            step_size=1
        ), 
        data=DataConfig(
            use_dataloader=True,
            training_batch_size=64, 
            fix_random_seed=False, 
        ),
        model=CIFARModelConfig(
            custom_act="relu",
            num_latent_layers=3, 
            latent_dims=[], 
            allow_residual=False,
            input_channels=3
        )
    )
    experiment = CIFARExperiment(config)
    experiment.train()
    output_labels = experiment.predict()