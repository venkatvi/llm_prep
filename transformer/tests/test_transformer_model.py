"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for complete transformer model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest
from transformer.transformer_model import TransformerModel, TransformerEncoderLayer


class TestTransformerEncoderLayer:
    """Test suite for TransformerEncoderLayer class."""
    
    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        embed_dim = 64
        num_heads = 8
        ffn_latent_dim = 256
        
        layer = TransformerEncoderLayer(embed_dim, num_heads, ffn_latent_dim)
        
        assert layer.attn.embed_dim == embed_dim
        assert layer.attn.num_heads == num_heads
        assert layer.ffn.layer_1.out_features == ffn_latent_dim
        assert layer.norm_1.normalized_shape == (embed_dim,)
        assert layer.norm_2.normalized_shape == (embed_dim,)
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        embed_dim = 64
        num_heads = 8
        ffn_latent_dim = 256
        batch_size = 4
        seq_len = 16
        
        layer = TransformerEncoderLayer(embed_dim, num_heads, ffn_latent_dim)
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        output = layer(x)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_residual_connections(self):
        """Test that residual connections work properly."""
        embed_dim = 32
        num_heads = 4
        ffn_latent_dim = 128
        batch_size = 2
        seq_len = 8
        
        layer = TransformerEncoderLayer(embed_dim, num_heads, ffn_latent_dim)
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Forward pass
        output = layer(x)
        
        # Output should be different from input (due to transformations)
        assert not torch.allclose(output, x)
        
        # But should maintain similar magnitude due to residual connections
        input_norm = x.norm(dim=-1).mean()
        output_norm = output.norm(dim=-1).mean()
        assert abs(input_norm - output_norm) < input_norm * 0.5  # Within 50%


class TestTransformerModel:
    """Test suite for TransformerModel class."""
    
    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        input_dim = 8
        embed_dim = 32
        ffn_latent_dim = 128
        num_layers = 2
        num_heads = 4
        output_dim = 1
        
        model = TransformerModel(
            input_dim, embed_dim, ffn_latent_dim, 
            num_layers, num_heads, output_dim
        )
        
        assert model.input_proj.in_features == input_dim
        assert model.input_proj.out_features == embed_dim
        assert len(model.layers) == num_layers
        assert model.out_proj.in_features == embed_dim
        assert model.out_proj.out_features == output_dim
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        input_dim = 8
        embed_dim = 32
        ffn_latent_dim = 128
        num_layers = 2
        num_heads = 4
        output_dim = 1
        batch_size = 4
        seq_len = 16
        
        model = TransformerModel(
            input_dim, embed_dim, ffn_latent_dim, 
            num_layers, num_heads, output_dim
        )
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x)
        
        assert output.shape == (batch_size, output_dim)
    
    def test_forward_gradient_flow(self):
        """Test gradients flow through entire model."""
        input_dim = 8
        embed_dim = 32
        ffn_latent_dim = 128
        num_layers = 2
        num_heads = 4
        output_dim = 1
        batch_size = 2
        seq_len = 8
        
        model = TransformerModel(
            input_dim, embed_dim, ffn_latent_dim, 
            num_layers, num_heads, output_dim
        )
        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check gradients in model parameters
        for param in model.parameters():
            assert param.grad is not None
    
    def test_different_configurations(self):
        """Test model works with different configurations."""
        test_configs = [
            (4, 16, 64, 1, 2, 1),
            (8, 32, 128, 2, 4, 1),
            (16, 64, 256, 3, 8, 5),
        ]
        
        batch_size = 2
        seq_len = 10
        
        for input_dim, embed_dim, ffn_latent_dim, num_layers, num_heads, output_dim in test_configs:
            model = TransformerModel(
                input_dim, embed_dim, ffn_latent_dim, 
                num_layers, num_heads, output_dim
            )
            x = torch.randn(batch_size, seq_len, input_dim)
            output = model(x)
            assert output.shape == (batch_size, output_dim)
    
    def test_global_average_pooling(self):
        """Test global average pooling behavior."""
        input_dim = 8
        embed_dim = 32
        ffn_latent_dim = 128
        num_layers = 1
        num_heads = 4
        output_dim = 1
        batch_size = 2
        
        model = TransformerModel(
            input_dim, embed_dim, ffn_latent_dim, 
            num_layers, num_heads, output_dim
        )
        
        # Test with different sequence lengths
        for seq_len in [5, 10, 20]:
            x = torch.randn(batch_size, seq_len, input_dim)
            output = model(x)
            assert output.shape == (batch_size, output_dim)
    
    def test_deterministic_output(self):
        """Test model produces deterministic output given same input."""
        input_dim = 8
        embed_dim = 32
        ffn_latent_dim = 128
        num_layers = 2
        num_heads = 4
        output_dim = 1
        batch_size = 2
        seq_len = 8
        
        model = TransformerModel(
            input_dim, embed_dim, ffn_latent_dim, 
            num_layers, num_heads, output_dim
        )
        model.eval()  # Set to evaluation mode
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output1 = model(x)
        output2 = model(x)
        
        assert torch.allclose(output1, output2)
    
    def test_layer_stacking(self):
        """Test that multiple layers are properly stacked."""
        input_dim = 8
        embed_dim = 32
        ffn_latent_dim = 128
        num_layers = 3
        num_heads = 4
        output_dim = 1
        batch_size = 2
        seq_len = 8
        
        model = TransformerModel(
            input_dim, embed_dim, ffn_latent_dim, 
            num_layers, num_heads, output_dim
        )
        
        # Check that we have the right number of layers
        assert len(model.layers) == num_layers
        
        # Each layer should be a TransformerEncoderLayer
        for layer in model.layers:
            assert isinstance(layer, TransformerEncoderLayer)
    
    def test_zero_input(self):
        """Test model handles zero input gracefully."""
        input_dim = 8
        embed_dim = 32
        ffn_latent_dim = 128
        num_layers = 2
        num_heads = 4
        output_dim = 1
        batch_size = 2
        seq_len = 8
        
        model = TransformerModel(
            input_dim, embed_dim, ffn_latent_dim, 
            num_layers, num_heads, output_dim
        )
        x = torch.zeros(batch_size, seq_len, input_dim)
        
        output = model(x)
        
        assert output.shape == (batch_size, output_dim)
        assert torch.isfinite(output).all()
    
    def test_large_sequence_length(self):
        """Test model works with large sequence lengths."""
        input_dim = 8
        embed_dim = 32
        ffn_latent_dim = 128
        num_layers = 1
        num_heads = 4
        output_dim = 1
        batch_size = 1
        seq_len = 1000  # Large sequence
        
        model = TransformerModel(
            input_dim, embed_dim, ffn_latent_dim, 
            num_layers, num_heads, output_dim
        )
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x)
        
        assert output.shape == (batch_size, output_dim)
        assert torch.isfinite(output).all()
    
    def test_model_training_mode(self):
        """Test model behavior in training vs eval mode."""
        input_dim = 8
        embed_dim = 32
        ffn_latent_dim = 128
        num_layers = 2
        num_heads = 4
        output_dim = 1
        batch_size = 2
        seq_len = 8
        
        model = TransformerModel(
            input_dim, embed_dim, ffn_latent_dim, 
            num_layers, num_heads, output_dim
        )
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Training mode
        model.train()
        train_output = model(x)
        
        # Eval mode
        model.eval()
        eval_output = model(x)
        
        # Outputs should be similar (no dropout in this implementation)
        assert torch.allclose(train_output, eval_output, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])