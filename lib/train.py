"""
Copyright (c) 2025. All rights reserved.
"""

"""
Training utilities for regression models.

This module provides training loops, data splitting, and optimization utilities
for training regression models with PyTorch. Includes support for train/validation
splits, various optimizers, and learning rate schedulers.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from lib.logger import Logger


@dataclass
class TrainContext:
    """
    Configuration container for training parameters.

    Holds all the necessary components and settings for training a regression model,
    including optimizer, scheduler, logging configuration, and training hyperparameters.

    Attributes:
        epochs (int): Number of training epochs
        optimizer (torch.optim.Optimizer): PyTorch optimizer instance
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
        loss_criterion (torch.nn.Module): Loss function for computing training loss
        tensorboard_log_dir (str): Directory for TensorBoard logs
        run_name (Optional[str]): Name for this training run (default: None)
        log_every_k_steps (int): Frequency of logging in epochs (default: 10)
    """

    epochs: int  # Number of training epochs
    optimizer: torch.optim.Optimizer  # PyTorch optimizer instance
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler  # Learning rate scheduler
    loss_criterion: torch.nn.Module  # Loss function
    tensorboard_log_dir: str  # Directory for TensorBoard logs
    run_name: Optional[str] = None  # Name for this training run
    log_every_k_steps: int = 10  # Frequency of logging (every k epochs)


def split_data(
    inputs: torch.Tensor, targets: torch.Tensor, val_split: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split dataset into training and validation sets with shuffling.

    Args:
        inputs (torch.Tensor): Input features tensor
        targets (torch.Tensor): Target values tensor
        val_split (float): Fraction of data to use for validation (default: 0.2)

    Returns:
        Tuple containing (train_inputs, train_targets, val_inputs, val_targets)
    """
    n_samples = inputs.size()[0]

    # Shuffle data randomly
    rand_indices = torch.randperm(n_samples)
    shuffled_inputs = inputs[rand_indices]
    shuffled_targets = targets[rand_indices]

    # Split into training and validation sets
    train_size: int = int((1 - val_split) * n_samples)
    train_inputs = shuffled_inputs[:train_size]
    train_targets = shuffled_targets[:train_size]
    val_inputs = shuffled_inputs[train_size:]
    val_targets = shuffled_targets[train_size:]

    return train_inputs, train_targets, val_inputs, val_targets


def train_model(
    model: torch.nn.Module, train_context: TrainContext, inputs: torch.Tensor, targets: torch.Tensor
) -> Tuple[float, float]:
    """
    Train a regression model using direct tensor processing.

    Args:
        model (torch.nn.Module): PyTorch model to train
        train_context (TrainContext): Configuration object with training parameters
        inputs (torch.Tensor): Input features for training
        targets (torch.Tensor): Target values for training

    Returns:
        Tuple[float, float]: Final (train_loss, val_loss) after training
    """
    logger = Logger(train_context.tensorboard_log_dir, train_context.run_name)

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
        if hasattr(train_context.lr_scheduler, "step"):
            if isinstance(train_context.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                train_context.lr_scheduler.step(loss.detach())
            else:
                train_context.lr_scheduler.step()

        # Validation phase
        if (epoch + 1) % train_context.log_every_k_steps == 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model(val_inputs)
                val_loss = train_context.loss_criterion(val_predictions, val_targets)

            current_lr = train_context.optimizer.param_groups[0]["lr"]
            logger.log_scalars(
                {
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item(),
                    "learning_rate": current_lr,
                },
                step=epoch + 1,
            )
            print(
                f"Epoch {epoch+1}/{train_context.epochs}, Train Loss: {loss.item():.4f}, "
                f"Val Loss: {val_loss.item():.4f}, LR: {current_lr:.6f}"
            )

    logger.close()

    # Return final losses
    model.eval()
    with torch.no_grad():
        final_train_predictions = model(train_inputs)
        final_train_loss = train_context.loss_criterion(final_train_predictions, train_targets)
        final_val_predictions = model(val_inputs)
        final_val_loss = train_context.loss_criterion(final_val_predictions, val_targets)

    return final_train_loss.item(), final_val_loss.item()


def train_model_with_dataloader(
    model: torch.nn.Module,
    train_context: TrainContext,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
) -> Tuple[float, float]:
    """
    Train a regression model using DataLoader for batch processing.

    Args:
        model (torch.nn.Module): PyTorch model to train
        train_context (TrainContext): Configuration object with training parameters
        train_dataloader (DataLoader): DataLoader for training data batches
        val_dataloader (DataLoader): DataLoader for validation data batches

    Returns:
        Tuple[float, float]: Final (train_loss, val_loss) after training
    """
    logger = Logger(train_context.tensorboard_log_dir, train_context.run_name)

    for epoch in range(train_context.epochs):
        # Training phase
        model.train()
        for batch_inputs, batch_targets in train_dataloader:
            # forward pass
            predictions = model(batch_inputs)
            loss = train_context.loss_criterion(predictions, batch_targets)

            # backward pass
            train_context.optimizer.zero_grad()
            loss.backward()
            train_context.optimizer.step()

        # Step the learning rate scheduler
        if hasattr(train_context.lr_scheduler, "step"):
            if isinstance(train_context.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                train_context.lr_scheduler.step(loss.detach())
            else:
                train_context.lr_scheduler.step()

        # Validation phase
        if (epoch + 1) % train_context.log_every_k_steps == 0:
            model.eval()
            with torch.no_grad():
                num_batches = 0
                val_loss = torch.rand(1, 1) * 0
                for batch_val_inputs, batch_val_targets in val_dataloader:
                    val_predictions = model(batch_val_inputs)
                    val_loss += train_context.loss_criterion(val_predictions, batch_val_targets)
                    num_batches += 1
                val_loss = val_loss / num_batches
            current_lr = train_context.optimizer.param_groups[0]["lr"]
            logger.log_scalars(
                {
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item(),
                    "learning_rate": current_lr,
                },
                step=epoch + 1,
            )
            print(
                f"Epoch {epoch+1}/{train_context.epochs}, Train Loss: {loss.item():.4f}, "
                f"Val Loss: {val_loss.item():.4f}, LR: {current_lr:.6f}"
            )

    # Calculate final losses over all data
    model.eval()
    final_train_loss = 0.0
    train_batches = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in train_dataloader:
            predictions = model(batch_inputs)
            final_train_loss += train_context.loss_criterion(predictions, batch_targets).item()
            train_batches += 1
    final_train_loss /= train_batches

    final_val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in val_dataloader:
            predictions = model(batch_inputs)
            final_val_loss += train_context.loss_criterion(predictions, batch_targets).item()
            val_batches += 1
    final_val_loss /= val_batches

    logger.close()

    return final_train_loss, final_val_loss


def train_encoder_decoder(
    model: torch.nn.Module,
    source_sequences: torch.Tensor,
    target_sequences: torch.Tensor,
    train_context: TrainContext,
) -> Tuple[float, float]:
    """Train encoder-decoder transformer model on sequence-to-sequence data.

    Handles training of encoder-decoder architectures by properly formatting input/output
    sequences for teacher forcing training. Creates decoder input (without last token)
    and decoder target (without first token) from target sequences.

    Args:
        model (torch.nn.Module): Encoder-decoder model to train
        source_sequences (torch.Tensor): Source sequences [batch_size, src_len, input_dim]
        target_sequences (torch.Tensor): Target sequences [batch_size, tgt_len, input_dim]
        train_context (TrainContext): Configuration object with training parameters

    Returns:
        Tuple[float, float]: Final (train_loss, val_loss) after training
    """
    logger = Logger(train_context.tensorboard_log_dir, train_context.run_name)

    # Split data into train and validation sets
    train_src_sequences, train_tgt_sequences, val_src_sequences, val_tgt_sequences = split_data(
        source_sequences, target_sequences
    )

    # source_sequences = [bs, src_len, embed_dim]
    # tgt_sequences = [bs, tgt_len, embed_dim]

    train_decoder_input = train_tgt_sequences[:, :-1, :]
    train_decoder_output = train_tgt_sequences[:, 1:, :]

    val_decoder_input = val_tgt_sequences[:, :-1, :]
    val_decoder_output = val_tgt_sequences[:, 1:, :]

    for epoch in range(train_context.epochs):
        model.train()

        predictions = model(train_src_sequences, train_decoder_input)
        loss = train_context.loss_criterion(predictions, train_decoder_output)

        # backward pass
        train_context.optimizer.zero_grad()
        loss.backward()
        train_context.optimizer.step()

        # Step the learning rate scheduler
        if hasattr(train_context.lr_scheduler, "step"):
            if isinstance(train_context.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                train_context.lr_scheduler.step(loss.detach())
            else:
                train_context.lr_scheduler.step()

        # Validation phase
        if (epoch + 1) % train_context.log_every_k_steps == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(val_src_sequences, val_decoder_input)
                val_loss = train_context.loss_criterion(predictions, val_decoder_output)
            current_lr = train_context.optimizer.param_groups[0]["lr"]
            logger.log_scalars(
                {
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item(),
                    "learning_rate": current_lr,
                },
                step=epoch + 1,
            )
            print(
                f"Epoch {epoch+1}/{train_context.epochs}, Train Loss: {loss.item():.4f}, "
                f"Val Loss: {val_loss.item():.4f}, LR: {current_lr:.6f}"
            )

    logger.close()

    # Return final losses
    model.eval()
    with torch.no_grad():
        final_train_predictions = model(train_src_sequences, train_decoder_input)
        final_train_loss = train_context.loss_criterion(
            final_train_predictions, train_decoder_output
        )
        final_val_predictions = model(val_src_sequences, val_decoder_input)
        final_val_loss = train_context.loss_criterion(final_val_predictions, val_decoder_output)

    return final_train_loss.item(), final_val_loss.item()


def predict_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    log_dir: str,
    run_name: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Generate predictions and calculate metrics on a dataset.

    Evaluates a trained model on the given dataset, computing predictions for each
    sample individually and calculating the overall Mean Squared Error (MSE).
    Logs results to TensorBoard and prints individual predictions.

    Args:
        model (torch.nn.Module): Trained PyTorch model in evaluation mode
        inputs (torch.Tensor): Input features for prediction [num_samples, ...]
        targets (torch.Tensor): True target values [num_samples, ...]
        log_dir (str): Directory for TensorBoard logs
        run_name (Optional[str]): Name for this prediction run

    Returns:
        List[torch.Tensor]: List of individual predictions as numpy arrays
    """
    logger = Logger(log_dir, run_name + "_predict")
    mse: torch.Tensor = torch.tensor(0.0)
    y_hat: List[torch.Tensor] = []

    # Set model to evaluation mode
    with torch.no_grad():
        for x, y in zip(inputs, targets):
            predictions = model(x)
            y_hat.append(predictions.numpy())
            mse += (predictions - y) ** 2
            print(f"Target: {y.item():.4f}, Actual: {predictions.item():.4f}")

        # Calculate and log mean squared error
        mse = mse / len(targets)
        logger.log_scalars({"MSE": mse.item()})
        print(f"MSE: {mse}")
    logger.close()
    return y_hat


def ar_predict(
    model: torch.nn.Module,
    input: torch.Tensor,
    num_steps_override: int,
    expanding_context: bool,
    max_seq_len: int,
    log_dir: str,
    run_name: Optional[str] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively from initial input sequence.

    Performs token-by-token generation using the trained autoregressive model.
    Supports both expanding context (keeping all previous tokens) and sliding
    window modes for different generation strategies.

    Args:
        model (torch.nn.Module): Trained autoregressive model with generate_next_token method
        input (torch.Tensor): Initial sequence [batch_size, seq_len, input_dim]
        num_steps_override (int): Number of generation steps to perform
        expanding_context (bool): If True, keep expanding context; if False, use sliding window
        max_seq_len (int): Maximum sequence length (for context truncation)
        log_dir (str): Directory for logging generation process
        run_name (Optional[str]): Name for this generation run

    Returns:
        torch.Tensor: Generated tokens [batch_size, num_generated_tokens, output_dim]
    """
    logger = Logger(log_dir, run_name + "_predict_autoregressively")
    generation_tokens: List[torch.Tensor] = []
    if num_steps_override is None:
        num_steps_override = 10  # Default number of steps

    current_input: torch.Tensor = input.clone()
    with torch.no_grad():
        for step in range(num_steps_override):
            print(f"Generating next_token in {step}, current_input size: {current_input.size()}")

            # Generate next token using the model's dedicated method
            next_token: torch.Tensor = model.generate_next_token(current_input)
            generation_tokens.append(next_token)

            # Update context based on decode configuration
            if expanding_context:
                # Keep expanding context (append new token)
                current_input = torch.cat(
                    [
                        current_input,
                        next_token,
                    ],
                    dim=1,
                )
                # Truncate if exceeds maximum sequence length
                if current_input.size(1) > max_seq_len:
                    current_input = current_input[:, -max_seq_len:, :]
            else:
                # Sliding window approach (remove first, add new)
                current_input = torch.cat(
                    [
                        current_input[:, 1:, :],  # Remove first token
                        next_token,  # Add generated token
                    ],
                    dim=1,
                )
    logger.close()
    return torch.cat(generation_tokens, dim=1)  # [bs, num_generated_tokens, output_dim]


def predict_encoder_decoder(
    model: torch.nn.Module,
    source_sequences: torch.Tensor,
    starting_target_token: torch.Tensor,
    num_steps: int,
    log_dir: str,
    run_name: Optional[str] = None,
) -> torch.Tensor:
    """Generate sequences using encoder-decoder model with autoregressive decoding.

    Encodes source sequences once, then autoregressively generates target tokens
    by feeding previous decoder outputs as input to the next decoding step.

    Args:
        model (torch.nn.Module): Trained encoder-decoder model
        source_sequences (torch.Tensor): Source sequences to encode [batch_size, src_len, input_dim]
        starting_target_token (torch.Tensor): Initial decoder token [batch_size, 1, input_dim]
        num_steps (int): Number of tokens to generate
        log_dir (str): Directory for logging
        run_name (Optional[str]): Name for this prediction run

    Returns:
        torch.Tensor: Generated sequence tokens [batch_size, num_steps, output_dim]
    """
    logger = Logger(log_dir, run_name + "_encoder_decoder_predict_ar")
    encoder_output = model.model.encode(source_sequences)

    decoder_input = starting_target_token  # [bs, 1, input_dim]

    generated_tokens: List[torch.Tensor] = []
    for _ in range(num_steps):
        decoder_output = model.model.decode(decoder_input, encoder_output)
        next_token = decoder_output[:, -1:, :]
        generated_tokens.append(next_token)

        decoder_input = torch.cat([decoder_input, next_token], dim=1)

    logger.close()
    return torch.cat(generated_tokens, dim=1)


def get_optimizer(optimizer_type: str, lr: float, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Factory function to create PyTorch optimizers.

    Args:
        optimizer_type (str): Type of optimizer ('adam', 'sgd', 'rmsprop')
        lr (float): Learning rate
        model (torch.nn.Module): Model whose parameters to optimize

    Returns:
        torch.optim.Optimizer: Configured optimizer instance

    Raises:
        ValueError: If optimizer_type is not supported
    """
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unsupported Optimizer Type: {optimizer_type}")


def get_lr_scheduler(
    lr_scheduler_type: str, optimizer: torch.optim.Optimizer, epochs: int, lr: float
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Factory function to create learning rate schedulers.

    Args:
        lr_scheduler_type (str): Type of scheduler ('steplr', 'exp', 'reduceonplat', 'cosine')
        optimizer (torch.optim.Optimizer): Optimizer to schedule
        epochs (int): Total number of training epochs
        lr (float): Initial learning rate (used for scheduler parameters)

    Returns:
        torch.optim.lr_scheduler.LRScheduler: Configured scheduler instance

    Raises:
        ValueError: If lr_scheduler_type is not supported
    """
    if lr_scheduler_type == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 10, gamma=lr / 100)
    elif lr_scheduler_type == "exp":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr / 100)
    elif lr_scheduler_type == "reduceonplat":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr / 100)
    elif lr_scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs / 100, eta_min=0.001
        )
    else:
        raise ValueError(f"Unsupported Learning Rate Scheduler Type: {lr_scheduler_type}")
