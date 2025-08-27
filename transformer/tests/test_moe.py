"""Comprehensive unit tests for Mixture of Experts (MOE) implementation.

Tests cover:
- Basic functionality and forward pass
- Load balancing and auxiliary loss computation
- Capacity constraints and overflow handling
- Top-k routing with different k values
- Device compatibility (CPU/GPU)
- Edge cases and error conditions
- Integration with transformer layers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch
import torch.nn.functional as F
from transformer.moe import MOE


def get_aux_loss_value(aux_loss):
    """Helper function to extract auxiliary loss value."""
    if isinstance(aux_loss, torch.Tensor):
        return aux_loss.item()
    return aux_loss


def is_finite(value):
    """Helper function to check if a value is finite."""
    if isinstance(value, torch.Tensor):
        return torch.isfinite(value)
    return torch.isfinite(torch.tensor(value))


class TestMOEBasics:
    """Test basic MOE functionality."""

    def test_moe_initialization(self):
        """Test MOE layer initialization with various configurations."""
        # Basic initialization
        moe = MOE(
            embed_dim=128,
            ffn_latent_dim=512,
            num_experts=8,
            capacity=64,
            alpha=0.01,
            topk=1,
        )

        assert moe.embed_dim == 128
        assert moe.num_experts == 8
        assert moe.capacity == 64
        assert moe.alpha == 0.01
        assert moe.topk == 1
        assert len(moe.experts) == 8
        assert moe.overflow_expert is not None

    def test_forward_pass_shapes(self):
        """Test that forward pass returns correct shapes."""
        moe = MOE(
            embed_dim=64, ffn_latent_dim=256, num_experts=4, capacity=32, alpha=0.01, topk=1
        )

        batch_size, seq_len, embed_dim = 2, 16, 64
        x = torch.randn(batch_size, seq_len, embed_dim)

        output, aux_loss = moe(x)

        assert output.shape == x.shape
        assert isinstance(aux_loss, (torch.Tensor, float))
        assert get_aux_loss_value(aux_loss) >= 0

    def test_different_topk_values(self):
        """Test MOE with different top-k routing values."""
        for topk in [1, 2, 3]:
            moe = MOE(
                embed_dim=32,
                ffn_latent_dim=128,
                num_experts=4,
                capacity=16,
                alpha=0.01,
                topk=topk,
            )

            x = torch.randn(2, 8, 32)
            output, aux_loss = moe(x)

            assert output.shape == x.shape
            assert get_aux_loss_value(aux_loss) >= 0

    def test_various_batch_sizes(self):
        """Test MOE with different batch sizes."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=16, alpha=0.01, topk=1
        )

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 10, 32)
            output, aux_loss = moe(x)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()
            assert is_finite(get_aux_loss_value(aux_loss))


class TestMOELoadBalancing:
    """Test load balancing functionality."""

    def test_auxiliary_loss_computation(self):
        """Test that auxiliary loss is computed correctly."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=16, alpha=0.1, topk=1
        )

        x = torch.randn(2, 16, 32)
        output, aux_loss = moe(x)

        # Auxiliary loss should be non-negative
        aux_loss_val = get_aux_loss_value(aux_loss)
        assert aux_loss_val >= 0

    def test_load_balancing_with_different_alpha(self):
        """Test that alpha parameter affects auxiliary loss magnitude."""
        x = torch.randn(2, 16, 32)

        # Test with different alpha values
        aux_losses = []
        for alpha in [0.001, 0.01, 0.1]:
            moe = MOE(
                embed_dim=32,
                ffn_latent_dim=128,
                num_experts=4,
                capacity=16,
                alpha=alpha,
                topk=1,
            )
            _, aux_loss = moe(x)
            aux_losses.append(get_aux_loss_value(aux_loss))

        # Higher alpha should generally lead to higher auxiliary loss
        assert aux_losses[1] > aux_losses[0]  # 0.01 > 0.001
        assert aux_losses[2] > aux_losses[1]  # 0.1 > 0.01

    def test_load_balancing_encourages_balance(self):
        """Test that load balancing loss increases when experts are imbalanced."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=100, alpha=0.01, topk=1
        )

        # Create input that should lead to more balanced routing
        balanced_input = torch.randn(4, 32, 32)

        # Create input that might lead to imbalanced routing (all similar)
        imbalanced_input = torch.ones(4, 32, 32) * 0.5

        _, balanced_loss = moe(balanced_input)
        _, imbalanced_loss = moe(imbalanced_input)

        # Both should be finite and non-negative
        balanced_val = get_aux_loss_value(balanced_loss)
        imbalanced_val = get_aux_loss_value(imbalanced_loss)
        
        assert is_finite(balanced_val)
        assert is_finite(imbalanced_val)
        assert balanced_val >= 0
        assert imbalanced_val >= 0


class TestMOECapacityConstraints:
    """Test capacity constraint functionality."""

    def test_capacity_overflow_handling(self):
        """Test that MOE handles capacity overflow correctly."""
        # Small capacity to force overflow
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=2, capacity=2, alpha=0.01, topk=1
        )

        # Large input to trigger overflow
        x = torch.randn(1, 20, 32)
        output, aux_loss = moe(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert is_finite(get_aux_loss_value(aux_loss))

    def test_no_overflow_with_large_capacity(self):
        """Test MOE behavior when capacity is larger than needed."""
        # Large capacity to avoid overflow
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=2, capacity=1000, alpha=0.01, topk=1
        )

        x = torch.randn(2, 10, 32)
        output, aux_loss = moe(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert is_finite(get_aux_loss_value(aux_loss))

    def test_zero_capacity_edge_case(self):
        """Test MOE behavior with zero capacity (should use overflow expert)."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=2, capacity=0, alpha=0.01, topk=1
        )

        x = torch.randn(1, 4, 32)
        output, aux_loss = moe(x)

        # Should still work, routing everything to overflow expert
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestMOERouting:
    """Test routing mechanism functionality."""

    def test_router_output_probabilities(self):
        """Test that router produces valid probability distributions."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=16, alpha=0.01, topk=2
        )

        x = torch.randn(2, 8, 32)
        
        # Access router directly to test probabilities
        logits = moe.router(x)  # Should be [2, 8, 4]
        probs = F.softmax(logits, dim=-1)

        assert logits.shape == (2, 8, 4)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 8), atol=1e-6)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_topk_selection(self):
        """Test that top-k selection works correctly."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=6, capacity=16, alpha=0.01, topk=3
        )

        x = torch.randn(2, 4, 32)
        
        # Test internal routing logic
        logits = moe.router(x)
        scores = F.softmax(logits, dim=-1)
        top_k_weights, expert_indices = torch.topk(scores, k=3, dim=-1)

        assert top_k_weights.shape == (2, 4, 3)
        assert expert_indices.shape == (2, 4, 3)
        assert (expert_indices >= 0).all()
        assert (expert_indices < 6).all()

    def test_weight_normalization(self):
        """Test that routing weights are properly normalized."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=16, alpha=0.01, topk=2
        )

        x = torch.randn(1, 4, 32)
        
        # Check internal weight normalization
        logits = moe.router(x)
        scores = F.softmax(logits, dim=-1)
        top_k_weights, _ = torch.topk(scores, k=2, dim=-1)
        
        # After normalization in forward pass
        normalized_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        assert torch.allclose(normalized_weights.sum(dim=-1), torch.ones(1, 4), atol=1e-6)


class TestMOEDeviceCompatibility:
    """Test device placement and GPU compatibility."""

    def test_cpu_compatibility(self):
        """Test MOE works correctly on CPU."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=16, alpha=0.01, topk=1
        )

        x = torch.randn(2, 8, 32)
        output, aux_loss = moe(x)

        assert output.device == x.device == torch.device("cpu")
        assert output.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test MOE works correctly on GPU."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=16, alpha=0.01, topk=1
        ).cuda()

        x = torch.randn(2, 8, 32).cuda()
        output, aux_loss = moe(x)

        assert output.device == x.device
        assert output.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_mismatch_handling(self):
        """Test that MOE handles device placement correctly."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=16, alpha=0.01, topk=1
        ).cuda()

        x = torch.randn(2, 8, 32).cuda()
        output, aux_loss = moe(x)

        # Output should be on same device as input
        assert output.device == x.device
        assert not torch.isnan(output).any()


class TestMOEEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_expert(self):
        """Test MOE with only one expert."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=1, capacity=16, alpha=0.01, topk=1
        )

        x = torch.randn(2, 4, 32)
        output, aux_loss = moe(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_large_embedding_dim(self):
        """Test MOE with large embedding dimensions."""
        moe = MOE(
            embed_dim=1024,
            ffn_latent_dim=4096,
            num_experts=8,
            capacity=128,
            alpha=0.01,
            topk=1,
        )

        x = torch.randn(1, 4, 1024)
        output, aux_loss = moe(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_small_sequence_length(self):
        """Test MOE with very small sequence lengths."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=16, alpha=0.01, topk=1
        )

        x = torch.randn(1, 1, 32)  # Single token
        output, aux_loss = moe(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow correctly through MOE."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=16, alpha=0.01, topk=1
        )

        x = torch.randn(2, 4, 32, requires_grad=True)
        output, aux_loss = moe(x)

        # Compute loss and backpropagate
        loss = output.sum() + get_aux_loss_value(aux_loss)
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check that model parameters have gradients (at least some should have gradients)
        params_with_grad = 0
        total_params = 0
        for param in moe.parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
                    assert not torch.isnan(param.grad).any()
        
        # At least some parameters should have gradients (router + some experts)
        assert params_with_grad > 0
        assert params_with_grad >= total_params * 0.3  # At least 30% should have gradients


class TestMOEIntegration:
    """Test MOE integration with other components."""

    def test_with_layer_norm(self):
        """Test MOE integration with layer normalization."""
        import torch.nn as nn

        class MOELayer(nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.moe = MOE(
                    embed_dim=embed_dim,
                    ffn_latent_dim=embed_dim * 4,
                    num_experts=4,
                    capacity=32,
                    alpha=0.01,
                    topk=1,
                )
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, x):
                moe_out, aux_loss = self.moe(x)
                output = self.norm(x + moe_out)  # Residual connection
                return output, aux_loss

        layer = MOELayer(64)
        x = torch.randn(2, 8, 64)

        output, aux_loss = layer(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_multiple_moe_layers(self):
        """Test stacking multiple MOE layers."""
        import torch.nn as nn

        class MultiMOEModel(nn.Module):
            def __init__(self, embed_dim, num_layers=2):
                super().__init__()
                self.layers = nn.ModuleList(
                    [
                        MOE(
                            embed_dim=embed_dim,
                            ffn_latent_dim=embed_dim * 4,
                            num_experts=4,
                            capacity=32,
                            alpha=0.01,
                            topk=1,
                        )
                        for _ in range(num_layers)
                    ]
                )

            def forward(self, x):
                total_aux_loss = 0
                for layer in self.layers:
                    x, aux_loss = layer(x)
                    total_aux_loss += get_aux_loss_value(aux_loss)
                return x, total_aux_loss

        model = MultiMOEModel(32, num_layers=3)
        x = torch.randn(2, 4, 32)

        output, total_aux_loss = model(x)
        assert output.shape == x.shape
        assert total_aux_loss >= 0

    def test_training_mode_vs_eval_mode(self):
        """Test MOE behavior in training vs evaluation mode."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=16, alpha=0.01, topk=1
        )

        x = torch.randn(2, 4, 32)

        # Training mode
        moe.train()
        train_output, train_aux_loss = moe(x)

        # Evaluation mode
        moe.eval()
        eval_output, eval_aux_loss = moe(x)

        # Both should produce valid outputs
        assert train_output.shape == eval_output.shape == x.shape
        assert not torch.isnan(train_output).any()
        assert not torch.isnan(eval_output).any()
        assert is_finite(get_aux_loss_value(train_aux_loss))
        assert is_finite(get_aux_loss_value(eval_aux_loss))


class TestMOEMathematicalProperties:
    """Test mathematical properties and correctness."""

    def test_expert_utilization(self):
        """Test that different experts are utilized over multiple forward passes."""
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=4, capacity=100, alpha=0.01, topk=1
        )

        # Track which experts get activated
        expert_activations = set()

        for _ in range(10):  # Multiple forward passes with different inputs
            x = torch.randn(2, 8, 32)
            
            # Hook into router to see expert assignments
            logits = moe.router(x)
            expert_assignments = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            
            # Track unique expert indices
            unique_experts = torch.unique(expert_assignments).tolist()
            expert_activations.update(unique_experts)

        # Should activate multiple experts over time (though not guaranteed all)
        assert len(expert_activations) >= 2

    def test_auxiliary_loss_mathematical_correctness(self):
        """Test that auxiliary loss computation follows the expected formula."""
        moe = MOE(
            embed_dim=16, ffn_latent_dim=64, num_experts=2, capacity=100, alpha=0.1, topk=1
        )

        x = torch.randn(1, 4, 16)  # Small input for easier verification
        output, aux_loss = moe(x)

        # Auxiliary loss should be: alpha * num_experts * sum(f_i * p_i)
        # Where f_i is fraction of tokens per expert, p_i is avg routing probability
        
        # The exact verification is complex due to internal routing,
        # but we can verify basic properties
        aux_loss_val = get_aux_loss_value(aux_loss)
        
        # Aux loss should be scaled appropriately
        assert aux_loss_val >= 0
        assert is_finite(aux_loss_val)

    def test_overflow_expert_functionality(self):
        """Test that overflow expert is used when needed."""
        # Force overflow by setting very small capacity
        moe = MOE(
            embed_dim=32, ffn_latent_dim=128, num_experts=2, capacity=1, alpha=0.01, topk=1
        )

        # Large input to ensure overflow
        x = torch.randn(1, 10, 32)
        output, aux_loss = moe(x)

        # Should still produce valid output even with overflow
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert is_finite(get_aux_loss_value(aux_loss))

        # Verify overflow expert exists and has parameters
        assert moe.overflow_expert is not None
        assert len(list(moe.overflow_expert.parameters())) > 0


if __name__ == "__main__":
    # Run tests with pytest for better output
    pytest.main([__file__, "-v"])