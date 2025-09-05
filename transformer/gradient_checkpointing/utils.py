"""
Copyright (c) 2025. All rights reserved.
"""

"""
Utilities for integrating gradient checkpointing with PyTorch's native checkpoint utilities.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Optional, Callable, Any, Tuple
import functools
from contextlib import contextmanager

from ..configs import CheckpointingConfig


class CheckpointWrapper(nn.Module):
    """Wrapper to apply PyTorch's checkpoint to any module."""
    
    def __init__(self, module: nn.Module, use_reentrant: bool = True):
        """Initialize checkpoint wrapper.
        
        Args:
            module: Module to wrap with checkpointing
            use_reentrant: Whether to use reentrant checkpointing
        """
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant
        
    def forward(self, *args, **kwargs):
        """Forward pass with checkpointing."""
        if self.training:
            return checkpoint.checkpoint(
                self.module, 
                *args, 
                use_reentrant=self.use_reentrant,
                **kwargs
            )
        else:
            return self.module(*args, **kwargs)


def selective_checkpoint(
    function: Callable,
    *args,
    condition: Optional[Callable[[], bool]] = None,
    use_reentrant: bool = True,
    **kwargs
) -> Any:
    """Apply checkpointing conditionally based on memory pressure or other conditions.
    
    Args:
        function: Function to potentially checkpoint
        *args: Arguments to pass to function
        condition: Optional condition function that returns True if checkpointing should be used
        use_reentrant: Whether to use reentrant checkpointing
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Result of function call, checkpointed or not based on condition
    """
    should_checkpoint = True
    
    if condition is not None:
        should_checkpoint = condition()
    
    if should_checkpoint and torch.is_grad_enabled():
        return checkpoint.checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
    else:
        return function(*args, **kwargs)


def memory_based_checkpoint_condition(threshold_mb: float = 1000.0) -> Callable[[], bool]:
    """Create a condition function that enables checkpointing based on memory usage.
    
    Args:
        threshold_mb: Memory threshold in MB below which checkpointing is enabled
        
    Returns:
        Condition function for selective checkpointing
    """
    def condition() -> bool:
        if not torch.cuda.is_available():
            return False
            
        available_memory_mb = (
            torch.cuda.get_device_properties(0).total_memory - 
            torch.cuda.memory_allocated()
        ) / (1024 ** 2)
        
        return available_memory_mb < threshold_mb
    
    return condition


class HybridCheckpointer:
    """Hybrid checkpointing that combines custom and PyTorch checkpoint utilities."""
    
    def __init__(self, config: CheckpointingConfig):
        """Initialize hybrid checkpointer.
        
        Args:
            config: Checkpointing configuration
        """
        self.config = config
        self.memory_condition = memory_based_checkpoint_condition(
            config.memory_threshold_mb
        )
        
    def checkpoint_function(
        self, 
        function: Callable,
        *args,
        force_custom: bool = False,
        **kwargs
    ) -> Any:
        """Checkpoint a function using hybrid strategy.
        
        Args:
            function: Function to checkpoint
            *args: Function arguments
            force_custom: Force use of custom checkpointing
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if not self.config.enable_checkpointing or not torch.is_grad_enabled():
            return function(*args, **kwargs)
            
        # Use custom checkpointing if forced or PyTorch checkpointing is disabled
        if force_custom or not self.config.use_pytorch_checkpoint:
            # For now, fall back to PyTorch checkpoint - custom implementation would go here
            return selective_checkpoint(
                function, 
                *args, 
                condition=self.memory_condition,
                **kwargs
            )
        else:
            # Use PyTorch's native checkpointing
            return selective_checkpoint(
                function,
                *args,
                condition=self.memory_condition if self.config.memory_threshold_mb > 0 else None,
                **kwargs
            )


@contextmanager
def checkpointing_context(config: CheckpointingConfig):
    """Context manager for configuring checkpointing behavior.
    
    Args:
        config: Checkpointing configuration
    """
    checkpointer = HybridCheckpointer(config)
    
    # Store original checkpoint function
    original_checkpoint = checkpoint.checkpoint
    
    def configured_checkpoint(function, *args, **kwargs):
        return checkpointer.checkpoint_function(function, *args, **kwargs)
    
    try:
        # Replace checkpoint function temporarily
        checkpoint.checkpoint = configured_checkpoint
        yield checkpointer
    finally:
        # Restore original checkpoint function
        checkpoint.checkpoint = original_checkpoint


def auto_checkpoint_layers(
    model: nn.Module,
    config: CheckpointingConfig,
    layer_names: Optional[list] = None
) -> nn.Module:
    """Automatically apply checkpointing to specified layers in a model.
    
    Args:
        model: Model to modify
        config: Checkpointing configuration  
        layer_names: List of layer names to checkpoint (None = auto-detect)
        
    Returns:
        Modified model with checkpointing applied
    """
    if not config.enable_checkpointing:
        return model
        
    if layer_names is None:
        # Auto-detect layers to checkpoint
        layer_names = []
        for name, module in model.named_modules():
            # Checkpoint FFN layers if configured
            if config.checkpoint_ffn and ('ffn' in name.lower() or 'mlp' in name.lower()):
                layer_names.append(name)
            # Checkpoint attention layers if configured  
            elif config.checkpoint_attention and ('attn' in name.lower() or 'attention' in name.lower()):
                layer_names.append(name)
                
    # Apply checkpointing to specified layers
    for name in layer_names:
        try:
            # Navigate to the module
            module = model
            for part in name.split('.'):
                module = getattr(module, part)
                
            # Wrap with checkpoint
            wrapped_module = CheckpointWrapper(module)
            
            # Replace in model
            parent = model
            parts = name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], wrapped_module)
            
        except AttributeError:
            print(f"Warning: Could not find layer {name} for checkpointing")
            
    return model


class CheckpointingProfiler:
    """Profiler for analyzing checkpointing performance."""
    
    def __init__(self):
        """Initialize profiler."""
        self.stats = {
            "forward_calls": 0,
            "backward_calls": 0,
            "recomputation_calls": 0,
            "memory_saved_mb": 0.0,
            "time_overhead_ms": 0.0
        }
        
    def reset(self):
        """Reset profiling statistics."""
        self.stats = {
            "forward_calls": 0,
            "backward_calls": 0,
            "recomputation_calls": 0,
            "memory_saved_mb": 0.0,
            "time_overhead_ms": 0.0
        }
        
    def get_stats(self) -> dict:
        """Get profiling statistics.
        
        Returns:
            Dictionary of profiling statistics
        """
        return self.stats.copy()
        
    @contextmanager
    def profile_checkpoint(self, operation_name: str = "checkpoint"):
        """Context manager for profiling checkpoint operations.
        
        Args:
            operation_name: Name of the operation being profiled
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        if start_time:
            start_time.record()
            
        try:
            yield
        finally:
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                
                # Update timing stats
                elapsed_ms = start_time.elapsed_time(end_time)
                self.stats["time_overhead_ms"] += elapsed_ms
                
            # Update memory stats
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                memory_saved = (start_memory - end_memory) / (1024 ** 2)
                if memory_saved > 0:
                    self.stats["memory_saved_mb"] += memory_saved


# Global profiler instance
checkpointing_profiler = CheckpointingProfiler()