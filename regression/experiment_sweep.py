"""
Copyright (c) 2025. All rights reserved.
"""

"""
Hyperparameter grid search utilities.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools

from lib.configs import DataConfig, ExperimentConfig, TrainConfig
from experiment import Experiment
from configs import RegressionModelConfig

if __name__ == "__main__":
    # Define parameter arrays for sweep
    epochs = [1000]
    custom_loss = ["mse", "huber"]
    optimizer = ["adam"] # sgd, rmsprop
    lr_scheduler = ["reduceonplat"] # steplr, exp
    lr = [0.001, 0.0001] # 0.01, 
    
    custom_act = ["relu",  "gelu"] #"tanh","leakyrelu", "silu"
    num_latent_layers = [5] # 1, 3
    latent_dims = [[16], [64, 128, 64], [128, 256, 512, 256, 128]]
    allow_residual = [True] # True
    
    use_dataloader = [False] #True
    training_batch_size = [32] #8, 16
    fix_random_seed = 42
    
    # Create cross product of all parameter combinations
    experiment_configs = []
    
    # Use itertools.product to generate all combinations
    for (epoch, loss, opt, lr_sched, learning_rate, act, layers, dims, residual, dataloader, batch_size) in itertools.product(
        epochs, custom_loss, optimizer, lr_scheduler, lr, 
        custom_act, num_latent_layers, latent_dims, allow_residual, 
        use_dataloader, training_batch_size
    ):
        # Skip invalid combinations where layer count doesn't match dims length
        if len(dims) != layers:
            continue
            
        # Create experiment configuration
        config = ExperimentConfig(
            type="nlinear",  # Use non-linear for sweep
            name=f"sweep_{opt}_{lr_sched}_{epoch}_{loss}_{act}_{layers}layers",
            train_config=TrainConfig(
                epochs=epoch,
                custom_loss=loss,
                optimizer=opt,
                lr_scheduler=lr_sched,
                lr=learning_rate,
                step_size=10,
            ),
            data=DataConfig(
                use_dataloader=dataloader,
                training_batch_size=batch_size,
                fix_random_seed=fix_random_seed
            ),
            model=RegressionModelConfig(
                custom_act=act,
                num_latent_layers=layers,
                latent_dims=dims,
                allow_residual=residual
            )
        )
        experiment_configs.append(config)
    
    # Create list of Experiment objects
    experiments = [Experiment(config) for config in experiment_configs]
    
    print(f"Generated {len(experiments)} experiment configurations")
    
    # Run first K experiments
    num_experiments = 8
    for i, experiment in enumerate(experiments[:num_experiments]):  
        print(f"Running experiment {i+1}/{len(experiments[:num_experiments])}: {experiment.config.name}")
        experiment.train()
        y_hat = experiment.predict()
        experiment.plot_results(y_hat)


