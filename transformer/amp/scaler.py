"""
Copyright (c) 2025. All rights reserved.
"""

"""
Advanced Gradient Scaler for Automatic Mixed Precision Training.

This module provides a sophisticated gradient scaler that handles:
1. Gradient scaling and unscaling with dynamic adjustment
2. Loss scaling with adaptive adjustment logic
3. Inf/NaN detection and recovery mechanisms
4. Optimized scaler update frequency for training stability
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Any, Callable
import math
import time
from dataclasses import dataclass
from enum import Enum
import warnings


class ScalingStrategy(Enum):
    """Scaling strategy for gradient scaler."""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    CONSERVATIVE = "conservative"


@dataclass
class ScalerConfig:
    """Configuration for advanced gradient scaler.
    
    Attributes:
        init_scale: Initial scale factor
        growth_factor: Factor to multiply scale when stable
        backoff_factor: Factor to divide scale when inf/NaN detected
        growth_interval: Iterations between scale increases
        strategy: Scaling strategy to use
        max_scale: Maximum allowed scale factor
        min_scale: Minimum allowed scale factor
        stability_threshold: Number of stable iterations before increasing scale
        instability_threshold: Number of unstable iterations before strategy change
        recovery_factor: Factor for recovery after instability
        enable_dynamic_frequency: Enable dynamic update frequency adjustment
        verbose: Enable verbose logging
    """
    init_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    max_scale: float = 2**24
    min_scale: float = 1.0
    stability_threshold: int = 1000
    instability_threshold: int = 10
    recovery_factor: float = 0.1
    enable_dynamic_frequency: bool = True
    verbose: bool = False


class GradientScaler:
    """Advanced gradient scaler with adaptive strategies and stability optimization.
    
    This scaler provides sophisticated gradient scaling capabilities including:
    - Multiple scaling strategies (fixed, dynamic, adaptive, conservative)
    - Inf/NaN detection and recovery
    - Dynamic update frequency adjustment
    - Training stability optimization
    - Comprehensive statistics and monitoring
    """
    
    def __init__(self, config: Optional[ScalerConfig] = None):
        """Initialize gradient scaler.
        
        Args:
            config: Scaler configuration, uses default if None
        """
        self.config = config or ScalerConfig()
        
        # Core scaling state
        self._scale = torch.tensor(self.config.init_scale, dtype=torch.float32)
        self._growth_tracker = 0
        self._current_step = 0
        
        # Stability tracking
        self._stable_iterations = 0
        self._unstable_iterations = 0
        self._consecutive_overflows = 0
        self._total_overflows = 0
        
        # Dynamic frequency adjustment
        self._current_growth_interval = self.config.growth_interval
        self._last_update_step = 0
        self._overflow_history = []
        self._max_history_size = 100
        
        # Statistics
        self.stats = {
            "total_steps": 0,
            "successful_steps": 0,
            "overflow_steps": 0,
            "scale_updates": 0,
            "scale_reductions": 0,
            "scale_increases": 0,
            "recovery_events": 0,
            "strategy_changes": 0
        }
        
        # Recovery state
        self._in_recovery_mode = False
        self._recovery_start_step = 0
        self._recovery_duration = 500
        
    def scale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scale a tensor (typically loss) by current scale factor.
        
        Args:
            tensor: Tensor to scale
            
        Returns:
            Scaled tensor
        """
        return tensor * self._scale.to(tensor.device)
        
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients in optimizer.
        
        Args:
            optimizer: Optimizer containing gradients to unscale
        """
        # Check if already unscaled
        if not getattr(optimizer, '_amp_stash', {}).get('already_unscaled', False):
            inv_scale = 1.0 / self._scale
            
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        param.grad.mul_(inv_scale)
            
            # Mark as unscaled
            if not hasattr(optimizer, '_amp_stash'):
                optimizer._amp_stash = {}
            optimizer._amp_stash['already_unscaled'] = True
            
    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step optimizer with gradient scaling handling.
        
        Args:
            optimizer: Optimizer to step
            
        Returns:
            True if step was taken, False if skipped due to inf/NaN
        """
        self.stats["total_steps"] += 1
        self._current_step += 1
        
        # Check for inf/NaN gradients
        has_inf_or_nan = self._check_inf_nan_gradients(optimizer)
        
        if has_inf_or_nan:
            # Skip optimizer step and handle overflow
            self._handle_overflow()
            self._clear_gradients(optimizer)
            return False
        else:
            # Take optimizer step
            optimizer.step()
            self.stats["successful_steps"] += 1
            self._stable_iterations += 1
            self._consecutive_overflows = 0
            
            # Mark as successful for dynamic frequency
            self._overflow_history.append(False)
            if len(self._overflow_history) > self._max_history_size:
                self._overflow_history.pop(0)
                
            return True
            
    def update(self) -> None:
        """Update scale factor based on current strategy and stability."""
        self._update_scale()
        self._update_strategy_if_needed()
        self._adjust_update_frequency()
        
        # Reset unscaled flag for next iteration
        # This would be set by the training loop's optimizer
        
    def _check_inf_nan_gradients(self, optimizer: torch.optim.Optimizer) -> bool:
        """Check if gradients contain inf or NaN values.
        
        Args:
            optimizer: Optimizer to check gradients
            
        Returns:
            True if inf/NaN detected, False otherwise
        """
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        return True
        return False
        
    def _handle_overflow(self) -> None:
        """Handle gradient overflow by reducing scale and updating statistics."""
        self.stats["overflow_steps"] += 1
        self._total_overflows += 1
        self._consecutive_overflows += 1
        self._unstable_iterations += 1
        self._stable_iterations = 0
        
        # Add to overflow history
        self._overflow_history.append(True)
        if len(self._overflow_history) > self._max_history_size:
            self._overflow_history.pop(0)
            
        # Reduce scale
        old_scale = self._scale.item()
        self._scale = torch.maximum(
            self._scale * self.config.backoff_factor,
            torch.tensor(self.config.min_scale)
        )
        self.stats["scale_reductions"] += 1
        
        if self.config.verbose:
            print(f"Overflow detected at step {self._current_step}. "
                  f"Scale reduced from {old_scale:.1f} to {self._scale.item():.1f}")
            
        # Check if we need recovery mode
        if self._consecutive_overflows >= 3:
            self._enter_recovery_mode()
            
    def _clear_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Clear gradients in optimizer."""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.zero_()
                    
        # Reset unscaled flag
        if hasattr(optimizer, '_amp_stash'):
            optimizer._amp_stash['already_unscaled'] = False
            
    def _update_scale(self) -> None:
        """Update scale factor based on current strategy."""
        if self.config.strategy == ScalingStrategy.FIXED:
            # No scale updates in fixed mode
            return
            
        elif self.config.strategy == ScalingStrategy.DYNAMIC:
            self._update_scale_dynamic()
            
        elif self.config.strategy == ScalingStrategy.ADAPTIVE:
            self._update_scale_adaptive()
            
        elif self.config.strategy == ScalingStrategy.CONSERVATIVE:
            self._update_scale_conservative()
            
    def _update_scale_dynamic(self) -> None:
        """Standard dynamic scaling update."""
        self._growth_tracker += 1
        
        # Increase scale if we've been stable long enough
        if (self._growth_tracker >= self._current_growth_interval and 
            self._consecutive_overflows == 0):
            
            old_scale = self._scale.item()
            self._scale = torch.minimum(
                self._scale * self.config.growth_factor,
                torch.tensor(self.config.max_scale)
            )
            
            if self._scale.item() > old_scale:
                self.stats["scale_increases"] += 1
                self._growth_tracker = 0
                
                if self.config.verbose:
                    print(f"Scale increased from {old_scale:.1f} to {self._scale.item():.1f}")
                    
    def _update_scale_adaptive(self) -> None:
        """Adaptive scaling based on recent overflow history."""
        if len(self._overflow_history) < 10:
            return
            
        recent_overflow_rate = sum(self._overflow_history[-20:]) / min(20, len(self._overflow_history))
        
        if recent_overflow_rate > 0.1:  # > 10% overflow rate
            # Reduce scale more aggressively
            self._scale *= 0.8
            self._scale = torch.maximum(self._scale, torch.tensor(self.config.min_scale))
        elif recent_overflow_rate < 0.01 and self._stable_iterations > self.config.stability_threshold:
            # Increase scale cautiously
            self._scale = torch.minimum(
                self._scale * 1.1,
                torch.tensor(self.config.max_scale)
            )
            self.stats["scale_increases"] += 1
            
    def _update_scale_conservative(self) -> None:
        """Conservative scaling that prioritizes stability."""
        # Only increase scale if we've been very stable
        if (self._stable_iterations > self.config.stability_threshold * 2 and 
            self._consecutive_overflows == 0 and
            self._total_overflows < 5):
            
            old_scale = self._scale.item()
            self._scale = torch.minimum(
                self._scale * 1.05,  # Very small increases
                torch.tensor(self.config.max_scale)
            )
            
            if self._scale.item() > old_scale:
                self.stats["scale_increases"] += 1
                self._stable_iterations = 0
                
    def _enter_recovery_mode(self) -> None:
        """Enter recovery mode after consecutive overflows."""
        if not self._in_recovery_mode:
            self._in_recovery_mode = True
            self._recovery_start_step = self._current_step
            
            # Drastically reduce scale
            self._scale *= self.config.recovery_factor
            self._scale = torch.maximum(self._scale, torch.tensor(self.config.min_scale))
            
            self.stats["recovery_events"] += 1
            
            if self.config.verbose:
                print(f"Entering recovery mode at step {self._current_step}. "
                      f"Scale set to {self._scale.item():.1f}")
                      
    def _exit_recovery_mode(self) -> None:
        """Exit recovery mode after stability is restored."""
        self._in_recovery_mode = False
        self._consecutive_overflows = 0
        
        if self.config.verbose:
            print(f"Exiting recovery mode at step {self._current_step}")
            
    def _update_strategy_if_needed(self) -> None:
        """Update scaling strategy based on training stability."""
        # Check if we should exit recovery mode
        if (self._in_recovery_mode and 
            self._current_step - self._recovery_start_step > self._recovery_duration and
            self._consecutive_overflows == 0):
            self._exit_recovery_mode()
            
        # Switch to conservative mode if too many overflows
        if (self.config.strategy != ScalingStrategy.CONSERVATIVE and
            self._unstable_iterations > self.config.instability_threshold):
            
            old_strategy = self.config.strategy
            self.config.strategy = ScalingStrategy.CONSERVATIVE
            self.stats["strategy_changes"] += 1
            
            if self.config.verbose:
                print(f"Switching from {old_strategy.value} to conservative scaling "
                      f"due to instability at step {self._current_step}")
                      
    def _adjust_update_frequency(self) -> None:
        """Dynamically adjust update frequency based on stability."""
        if not self.config.enable_dynamic_frequency:
            return
            
        if len(self._overflow_history) < 20:
            return
            
        recent_overflow_rate = sum(self._overflow_history[-20:]) / 20
        
        if recent_overflow_rate > 0.05:  # High overflow rate
            # Increase growth interval (less frequent increases)
            self._current_growth_interval = min(
                self._current_growth_interval * 1.5,
                self.config.growth_interval * 4
            )
        elif recent_overflow_rate < 0.01:  # Very stable
            # Decrease growth interval (more frequent increases)
            self._current_growth_interval = max(
                self._current_growth_interval * 0.9,
                self.config.growth_interval * 0.5
            )
            
    def get_scale(self) -> float:
        """Get current scale factor.
        
        Returns:
            Current scale factor as float
        """
        return self._scale.item()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics.
        
        Returns:
            Dictionary containing scaling statistics
        """
        stats = self.stats.copy()
        stats.update({
            "current_scale": self.get_scale(),
            "current_strategy": self.config.strategy.value,
            "current_growth_interval": self._current_growth_interval,
            "stable_iterations": self._stable_iterations,
            "unstable_iterations": self._unstable_iterations,
            "consecutive_overflows": self._consecutive_overflows,
            "total_overflows": self._total_overflows,
            "in_recovery_mode": self._in_recovery_mode,
            "success_rate": (
                self.stats["successful_steps"] / max(1, self.stats["total_steps"])
            ),
            "overflow_rate": (
                self.stats["overflow_steps"] / max(1, self.stats["total_steps"])
            )
        })
        
        if len(self._overflow_history) > 0:
            stats["recent_overflow_rate"] = sum(self._overflow_history) / len(self._overflow_history)
            
        return stats
        
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = {
            "total_steps": 0,
            "successful_steps": 0,
            "overflow_steps": 0,
            "scale_updates": 0,
            "scale_reductions": 0,
            "scale_increases": 0,
            "recovery_events": 0,
            "strategy_changes": 0
        }
        self._stable_iterations = 0
        self._unstable_iterations = 0
        self._consecutive_overflows = 0
        self._total_overflows = 0
        self._overflow_history.clear()
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scaler state from dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        self._scale = torch.tensor(state_dict["scale"], dtype=torch.float32)
        self._growth_tracker = state_dict["growth_tracker"]
        self._current_step = state_dict["current_step"]
        self.stats = state_dict["stats"]
        
    def state_dict(self) -> Dict[str, Any]:
        """Get scaler state as dictionary.
        
        Returns:
            State dictionary
        """
        return {
            "scale": self._scale.item(),
            "growth_tracker": self._growth_tracker,
            "current_step": self._current_step,
            "stats": self.stats
        }


class AMPTrainingLoop:
    """Training loop wrapper with advanced AMP and gradient scaling."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler_config: Optional[ScalerConfig] = None,
        enable_autocast: bool = True
    ):
        """Initialize AMP training loop.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            scaler_config: Configuration for gradient scaler
            enable_autocast: Whether to use autocast for mixed precision
        """
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradientScaler(scaler_config)
        self.enable_autocast = enable_autocast
        
        # Training statistics
        self.training_stats = {
            "epochs_completed": 0,
            "batches_processed": 0,
            "total_loss": 0.0,
            "overflow_batches": 0
        }
        
    def train_step(
        self,
        batch_data: Any,
        criterion: nn.Module,
        forward_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Perform a single training step with AMP.
        
        Args:
            batch_data: Input batch data
            criterion: Loss criterion
            forward_fn: Optional custom forward function
            
        Returns:
            Dictionary containing step results
        """
        self.optimizer.zero_grad()
        
        # Default forward function
        if forward_fn is None:
            def forward_fn(data):
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    inputs, targets = data
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    return loss, outputs
                else:
                    return self.model(data), None
                    
        # Forward pass with optional autocast
        if self.enable_autocast:
            with torch.cuda.amp.autocast():
                result = forward_fn(batch_data)
                if isinstance(result, tuple):
                    loss, outputs = result
                else:
                    loss, outputs = result, None
        else:
            result = forward_fn(batch_data)
            if isinstance(result, tuple):
                loss, outputs = result
            else:
                loss, outputs = result, None
                
        # Scale loss and backward pass
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        
        # Unscale gradients for clipping
        self.scaler.unscale_(self.optimizer)
        
        # Optional gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Step optimizer and update scaler
        step_successful = self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update training statistics
        self.training_stats["batches_processed"] += 1
        if step_successful:
            self.training_stats["total_loss"] += loss.item()
        else:
            self.training_stats["overflow_batches"] += 1
            
        return {
            "loss": loss.item() if step_successful else float('inf'),
            "scaled_loss": scaled_loss.item(),
            "step_successful": step_successful,
            "grad_norm": grad_norm.item() if step_successful else float('inf'),
            "current_scale": self.scaler.get_scale(),
            "outputs": outputs
        }
        
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive training and scaling statistics.
        
        Returns:
            Combined statistics dictionary
        """
        stats = {
            "training": self.training_stats.copy(),
            "scaler": self.scaler.get_stats()
        }
        
        # Calculate derived metrics
        if self.training_stats["batches_processed"] > 0:
            stats["training"]["avg_loss"] = (
                self.training_stats["total_loss"] / 
                max(1, self.training_stats["batches_processed"] - self.training_stats["overflow_batches"])
            )
            stats["training"]["overflow_rate"] = (
                self.training_stats["overflow_batches"] / self.training_stats["batches_processed"]
            )
            
        return stats


def create_optimized_scaler_config(
    training_instability: str = "medium",
    model_size: str = "medium",
    precision_requirements: str = "balanced"
) -> ScalerConfig:
    """Create optimized scaler configuration based on training characteristics.
    
    Args:
        training_instability: Level of training instability ("low", "medium", "high")
        model_size: Model size category ("small", "medium", "large") 
        precision_requirements: Precision requirements ("speed", "balanced", "precision")
        
    Returns:
        Optimized scaler configuration
    """
    if training_instability == "low":
        return ScalerConfig(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1000,
            strategy=ScalingStrategy.DYNAMIC
        )
    elif training_instability == "medium":
        return ScalerConfig(
            init_scale=2**14,
            growth_factor=1.5,
            backoff_factor=0.6,
            growth_interval=1500,
            strategy=ScalingStrategy.ADAPTIVE,
            enable_dynamic_frequency=True
        )
    else:  # high instability
        return ScalerConfig(
            init_scale=2**12,
            growth_factor=1.2,
            backoff_factor=0.7,
            growth_interval=2500,
            strategy=ScalingStrategy.CONSERVATIVE,
            stability_threshold=2000,
            recovery_factor=0.05
        )


if __name__ == "__main__":
    # Demo usage
    print("Advanced Gradient Scaler Demo")
    print("=" * 50)
    
    # Create test model and optimizer
    model = nn.Linear(256, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test different configurations
    configs = {
        "stable": create_optimized_scaler_config("low"),
        "unstable": create_optimized_scaler_config("high"),
        "adaptive": create_optimized_scaler_config("medium")
    }
    
    for name, config in configs.items():
        print(f"\nTesting {name} configuration:")
        scaler = GradientScaler(config)
        print(f"Initial scale: {scaler.get_scale()}")
        print(f"Strategy: {config.strategy.value}")
        print(f"Growth interval: {config.growth_interval}")