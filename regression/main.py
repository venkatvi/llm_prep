"""
Copyright (c) 2025. All rights reserved.
"""

"""
CLI for training regression models.
"""

import argparse
import os
import sys
from typing import List

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import (
    AutoregressiveDecodeConfig,
    EncoderDecoderConfig,
    RegressionModelConfig,
    TransformerModelConfig,
)
from experiment import EncoderDecoderExperiment, RegressionExperiment, TransformerExperiment

from lib.configs import DataConfig, ExperimentConfig, TrainConfig


def parse_latent_dims(value: str) -> List[int]:
    """Parse comma-separated string into integer list.

    Converts string like '128,64,32' into [128, 64, 32] for layer dimensions.

    Args:
        value (str): Comma-separated string of integers

    Returns:
        List[int]: List of parsed integer values

    Raises:
        ValueError: If any value in the string cannot be converted to int
    """
    return [int(x.strip()) for x in value.split(",")]


if __name__ == "__main__":
    # Set up command line argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train a regression model"
    )
    # Experiment name
    parser.add_argument(
        "--type",
        type=str,
        default="linear",
        help="Type of regression model - linear or non-linear. Default is linear regression.",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name for this training run in TensorBoard logs."
    )
    # Train Loop
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs to train a model. Default is 1000.",
    )
    parser.add_argument(
        "--custom_loss",
        type=str,
        default="mse",
        help="Custom Loss function to use for training loop.",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Type of optimizer to use. Default is Adam."
    )
    ## Optimizer and Learning Rate
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate for training. Default is 0.01"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="reduceonplat",
        help="LR scheduler to get better performance",
    )
    # Data
    parser.add_argument(
        "--use_dataloader",
        action="store_true",
        help="Use dataloader to iterate on data instead of large torch tensors.",
    )
    parser.add_argument(
        "--training_batch_size",
        type=int,
        default=8,
        help="Number of training samples per batch to iterate over loss computation.",
    )
    parser.add_argument(
        "--fix_random_seed",
        action="store_true",
        help="Fix random data gen seed for finding consistencies between runs.",
    )
    # Model
    parser.add_argument(
        "--custom_act",
        type=str,
        default="relu",
        help="Custom activation function to be enabled. Default is ReLU.",
    )
    parser.add_argument(
        "--num_latent_layers",
        type=int,
        default=1,
        help="Number of latent layers to use in non linear regression model. Default is 1.",
    )
    parser.add_argument(
        "--latent_dims",
        type=parse_latent_dims,
        default=[256],
        help="Latent dimension for latent layer in non-linear regression models. Uses comma-separated values (e.g., '128,64,32'). Default is 256. Number of latent dims should match the number of latent layers.",
    )
    parser.add_argument(
        "--allow_residual",
        action="store_true",
        help="Allow residual connections after activations in non linear reg model.",
    )
    parser.add_argument(
        "--autoregressive", action="store_true", help="Run transformer in autoregressive mode."
    )
    parser.add_argument(
        "--encoderdecoder",
        action="store_true",
        help="Use encoder-decoder like transformers for seq2seq translation",
    )
    args: argparse.Namespace = parser.parse_args()

    regression_model_config: RegressionModelConfig = RegressionModelConfig(
        name=args.type,
        custom_act=args.custom_act,
        num_latent_layers=args.num_latent_layers,
        latent_dims=args.latent_dims,
        allow_residual=args.allow_residual,
    )
    # preset transformer config
    transformer_model_config: TransformerModelConfig = TransformerModelConfig(
        name=args.type,
        max_seq_len=128,
        input_dim=1,
        embed_dim=64,
        ffn_latent_dim=128,
        num_layers=2,
        num_heads=2,
        output_dim=1,
        apply_causal_mask=True,
        autoregressive_mode=True,
        decode_config=AutoregressiveDecodeConfig(
            num_steps=10, expanding_context=True, max_seq_len=40
        ),
    )
    encoder_decoder_config: EncoderDecoderConfig = EncoderDecoderConfig(
        name=args.type,
        max_seq_len=128,
        input_dim=1,
        embed_dim=64,
        ffn_latent_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        output_dim=1,
        apply_causal_mask=True,
        autoregressive_mode=True,
        decode_config=AutoregressiveDecodeConfig(
            num_steps=10, expanding_context=True, max_seq_len=40
        ),
    )
    if args.type == "transformer":
        if args.encoderdecoder:
            model_config = encoder_decoder_config
        else:
            model_config = transformer_model_config
    else:
        model_config = regression_model_config
    experiment_config: ExperimentConfig = ExperimentConfig(
        type=args.type,
        name=args.run_name,
        train_config=TrainConfig(
            epochs=args.epochs,
            custom_loss=args.custom_loss,
            optimizer=args.optimizer,
            lr_scheduler=args.lr_scheduler,
            lr=args.lr,
            step_size=10,
        ),
        data=DataConfig(
            use_dataloader=args.use_dataloader,
            training_batch_size=args.training_batch_size,
            fix_random_seed=args.fix_random_seed,
        ),
        model=model_config,
    )

    if args.type == "transformer":
        if args.encoderdecoder:
            experiment: EncoderDecoderExperiment = EncoderDecoderExperiment(experiment_config)

            # Train
            experiment.train()

            # Generate sequences
            generated_tokens: torch.Tensor = experiment.predict_encoder_decoder()

        else:
            experiment: TransformerExperiment = TransformerExperiment(
                experiment_config, args.autoregressive
            )
            input: torch.Tensor
            targets: torch.Tensor
            input, targets = experiment.model.generate_data(random_seed=32)

            experiment.train()

            if args.autoregressive:
                generated_tokens: torch.Tensor = experiment.predict_autoregressively(input)
            else:
                y_hat: torch.Tensor = experiment.predict()

    else:
        experiment: RegressionExperiment = RegressionExperiment(experiment_config)

        # Train
        experiment.train()

        # Generate predictions on full dataset
        y_hat: torch.Tensor = experiment.predict()

        # Create and log visualization plot
        experiment.plot_results(y_hat)
