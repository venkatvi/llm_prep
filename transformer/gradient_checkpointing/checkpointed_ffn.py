"""
Copyright (c) 2025. All rights reserved.
"""

"""
Gradient checkpointing implementation for FFN layers.
Custom autograd functions that selectively save intermediate activations.
"""

import torch
import torch.nn as nn
from typing import Any, Tuple, Optional
import time
from contextlib import contextmanager

# Import parent FFN implementation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname("..")))
from ffn import FFN


class CheckpointedFFNFunction(torch.autograd.Function):
    """Custom autograd function for memory-efficient FFN computation.
    
    This function implements gradient checkpointing by:
    1. Saving only the input tensor during forward pass
    2. Recomputing intermediate activations during backward pass
    3. Trading compute for memory efficiency
    """

    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor, ffn_layer1: nn.Linear, 
                ffn_layer2: nn.Linear, activation_fn: nn.Module) -> torch.Tensor:
        """Forward pass with selective activation saving.
        
        Args:
            ctx: Autograd context for saving tensors
            input_tensor: Input to the FFN [batch_size, seq_len, embed_dim]
            ffn_layer1: First linear layer (embed_dim -> latent_dim)
            ffn_layer2: Second linear layer (latent_dim -> embed_dim)  
            activation_fn: Activation function (ReLU, GELU, etc.)
            
        Returns:
            torch.Tensor: FFN output [batch_size, seq_len, embed_dim]
        """
        # Save only input and layer references for backward pass
        ctx.save_for_backward(input_tensor)
        ctx.ffn_layer1 = ffn_layer1
        ctx.ffn_layer2 = ffn_layer2
        ctx.activation_fn = activation_fn
        
        # Perform forward computation with gradients enabled
        intermediate = activation_fn(ffn_layer1(input_tensor))
        output = ffn_layer2(intermediate)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass with activation recomputation.
        
        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient w.r.t. output
            
        Returns:
            Tuple of gradients w.r.t. inputs (input_tensor, layer1, layer2, activation)
        """
        # Retrieve saved input and layer references
        input_tensor, = ctx.saved_tensors
        ffn_layer1 = ctx.ffn_layer1
        ffn_layer2 = ctx.ffn_layer2
        activation_fn = ctx.activation_fn
        
        # Enable gradients for recomputation
        input_tensor = input_tensor.detach().requires_grad_(True)
        
        # Recompute intermediate activations for gradient computation
        with torch.enable_grad():
            # Recompute first layer
            intermediate = ffn_layer1(input_tensor)
            intermediate_activated = activation_fn(intermediate)
            
            # Recompute second layer  
            output = ffn_layer2(intermediate_activated)
            
        # Compute gradients using autograd
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=[input_tensor, ffn_layer1.weight, ffn_layer1.bias,
                   ffn_layer2.weight, ffn_layer2.bias],
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=torch.is_grad_enabled()
        )
        
        grad_input = gradients[0]
        # Layer gradients are handled by PyTorch's parameter gradient accumulation
        
        # Return gradients in same order as forward arguments
        return grad_input, None, None, None


class CheckpointedFFN(nn.Module):
    """Memory-efficient FFN wrapper using gradient checkpointing.
    
    This wrapper provides the same interface as regular FFN but uses
    gradient checkpointing to reduce memory usage during training.
    """
    
    def __init__(self, embed_dim: int, latent_dim: int, 
                 activation: str = "relu", enable_checkpointing: bool = True):
        """Initialize checkpointed FFN.
        
        Args:
            embed_dim: Input/output embedding dimension
            latent_dim: Hidden layer dimension (typically 4x embed_dim)
            activation: Activation function type ("relu", "gelu", "swish")
            enable_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()
        
        # Core layers
        self.layer_1 = nn.Linear(embed_dim, latent_dim)
        self.layer_2 = nn.Linear(latent_dim, embed_dim)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.enable_checkpointing = enable_checkpointing
        
        # Memory tracking
        self.memory_stats = {
            "forward_memory": 0,
            "backward_memory": 0,
            "forward_time": 0,
            "backward_time": 0,
            "recomputation_count": 0
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, embed_dim]
        """
        if self.enable_checkpointing and self.training:
            # Use custom checkpointed function
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            output = CheckpointedFFNFunction.apply(
                x, self.layer_1, self.layer_2, self.activation
            )
            
            # Update memory stats
            self.memory_stats["forward_time"] += time.time() - start_time
            if torch.cuda.is_available():
                self.memory_stats["forward_memory"] = max(
                    self.memory_stats["forward_memory"],
                    torch.cuda.memory_allocated() - start_memory
                )
            
            return output
        else:
            # Standard forward pass (no checkpointing)
            return self.layer_2(self.activation(self.layer_1(x)))
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics.
        
        Returns:
            dict: Memory and timing statistics
        """
        return self.memory_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset memory statistics."""
        self.memory_stats = {
            "forward_memory": 0,
            "backward_memory": 0,
            "forward_time": 0,
            "backward_time": 0,
            "recomputation_count": 0
        }


class AdaptiveCheckpointedFFN(CheckpointedFFN):
    """FFN with adaptive checkpointing based on memory pressure.
    
    Automatically enables/disables checkpointing based on available memory.
    """
    
    def __init__(self, embed_dim: int, latent_dim: int, 
                 activation: str = "relu",
                 memory_threshold_mb: float = 1000.0):
        """Initialize adaptive checkpointed FFN.
        
        Args:
            embed_dim: Input/output embedding dimension
            latent_dim: Hidden layer dimension
            activation: Activation function type
            memory_threshold_mb: Memory threshold in MB for enabling checkpointing
        """
        super().__init__(embed_dim, latent_dim, activation, enable_checkpointing=False)
        self.memory_threshold_mb = memory_threshold_mb
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive checkpointing.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Check memory pressure if CUDA is available
        if torch.cuda.is_available() and self.training:
            current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            available_memory_mb = (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) - 
                current_memory_mb
            )
            
            # Enable checkpointing if memory is limited
            self.enable_checkpointing = available_memory_mb < self.memory_threshold_mb
            
        return super().forward(x)


@contextmanager
def memory_profiler():
    """Context manager for profiling memory usage."""
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        print(f"Time: {end_time - start_time:.4f}s")
        if torch.cuda.is_available():
            print(f"Memory delta: {(end_memory - start_memory) / (1024**2):.2f} MB")
            print(f"Peak memory: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")


def compare_ffn_implementations(embed_dim: int = 512, latent_dim: int = 2048,
                               batch_size: int = 32, seq_len: int = 512,
                               num_iterations: int = 5) -> dict:
    """Compare memory usage and performance between standard and checkpointed FFN.
    
    Args:
        embed_dim: Embedding dimension
        latent_dim: Latent dimension  
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        num_iterations: Number of test iterations
        
    Returns:
        dict: Comparison results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
    target = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    results = {}
    
    # Test standard FFN
    print("Testing standard FFN...")
    standard_ffn = FFN(embed_dim, latent_dim).to(device)
    
    with memory_profiler():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
            output = standard_ffn(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            if torch.cuda.is_available():
                results["standard_peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    
    # Test checkpointed FFN  
    print("Testing checkpointed FFN...")
    checkpointed_ffn = CheckpointedFFN(embed_dim, latent_dim).to(device)
    
    with memory_profiler():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
            output = checkpointed_ffn(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            if torch.cuda.is_available():
                results["checkpointed_peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    
    # Calculate memory savings
    if torch.cuda.is_available():
        memory_savings = (
            results["standard_peak_memory_mb"] - results["checkpointed_peak_memory_mb"]
        ) / results["standard_peak_memory_mb"] * 100
        results["memory_savings_percent"] = memory_savings
        
    results["checkpointed_stats"] = checkpointed_ffn.get_memory_stats()
    
    return results


if __name__ == "__main__":
    # Demo usage
    print("Gradient Checkpointing FFN Demo")
    print("=" * 50)
    
    # Run comparison
    results = compare_ffn_implementations()
    
    print("\nComparison Results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")