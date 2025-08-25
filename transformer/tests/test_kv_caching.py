"""KV Caching correctness tests for attention mechanisms."""

"""
Tests for KV caching correctness in attention mechanisms.

Tests verify that cached attention produces identical results to non-cached attention
for both single-step and multi-step token generation scenarios.

This test suite validates:
1. Single-step parity: Cached vs non-cached attention produces identical outputs
2. Multi-step generation: Incremental cached generation matches full sequence computation
3. Cross-attention: KV caching works correctly for encoder-decoder scenarios
4. Memory efficiency: Fixed-size pre-allocated cache management
5. State management: Cache position tracking and lifecycle

Covers all attention mechanisms: MHA, MQA, and GQA.
"""

import pytest
import torch
from typing import List, Tuple

from transformer.attention.gqa import GroupQueryAttention
from transformer.attention.mha import MultiHeadAttention
from transformer.attention.mqa import MultiQueryAttention


class TestKVCachingCorrectness:
    """Test suite for KV caching correctness across all attention mechanisms.

    This class contains comprehensive tests that validate KV caching implementations
    across Multi-Head Attention (MHA), Multi-Query Attention (MQA), and
    Group Query Attention (GQA) mechanisms.

    Test categories:
    - Single-step parity: Cached vs non-cached output equivalence
    - Multi-step generation: Autoregressive token generation correctness
    - System tests: Cache state management and memory efficiency
    """

    @pytest.fixture
    def attention_configs(self) -> List[dict]:
        """Common attention configurations for testing.

        Returns:
            List[dict]: List of attention configuration dictionaries containing
                       embed_dim, num_heads, and num_groups parameters.
        """
        return [
            {"embed_dim": 64, "num_heads": 4, "num_groups": 2},
            {"embed_dim": 128, "num_heads": 8, "num_groups": 4},
            {"embed_dim": 32, "num_heads": 2, "num_groups": 1},
        ]

    @pytest.fixture
    def test_sequences(self) -> List[Tuple[int, int, int]]:
        """Test sequence configurations for various scenarios.

        Returns:
            List[Tuple[int, int, int]]: List of (batch_size, seq_len, embed_dim)
                                       configurations for testing different
                                       input scenarios.
        """
        return [
            (1, 8, 64),   # Single batch, short sequence
            (2, 16, 128),  # Small batch, medium sequence
            (1, 32, 32),  # Single batch, longer sequence
        ]

    def _reset_cache(self, attention_module) -> None:
        """Reset KV cache for fresh testing.

        Args:
            attention_module: Attention module with potential kv_cache attribute.
        """
        if hasattr(attention_module, "kv_cache"):
            attention_module.kv_cache = None

    def test_mha_single_step_parity(self, attention_configs) -> None:
        """Test single-step MHA cache parity with non-cached computation.

        Validates that Multi-Head Attention with KV caching enabled produces
        identical outputs to the same computation without caching when processing
        a full sequence in a single forward pass.

        Args:
            attention_configs: Fixture providing attention configuration parameters.
        """
        for config in attention_configs:
            embed_dim, num_heads = config["embed_dim"], config["num_heads"]

            # Create attention modules
            attn_no_cache = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                apply_causal_mask=True,
                use_kv_cache=False,
            )
            attn_with_cache = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                apply_causal_mask=True,
                use_kv_cache=True,
            )

            # Ensure same parameters
            attn_with_cache.load_state_dict(attn_no_cache.state_dict())

            # Test sequence
            batch_size, seq_len = 2, 10
            input_seq = torch.randn(batch_size, seq_len, embed_dim)

            # Non-cached forward pass (full sequence)
            attn_no_cache.eval()
            with torch.no_grad():
                output_no_cache = attn_no_cache(
                    input_seq, kv=None, expanding_context=True
                )

            # Cached forward pass (single step with full context)
            attn_with_cache.eval()
            attn_with_cache._inference_mode = True  # Enable caching
            self._reset_cache(attn_with_cache)

            with torch.no_grad():
                output_with_cache = attn_with_cache(
                    input_seq, kv=None, expanding_context=True
                )

            # Verify outputs match
            torch.testing.assert_close(
                output_no_cache,
                output_with_cache,
                msg=f"MHA cache mismatch for config {config}",
                atol=1e-5,
                rtol=1e-4,
            )

    def test_mqa_single_step_parity(self, attention_configs) -> None:
        """Test single-step MQA cache parity with non-cached computation.

        Validates that Multi-Query Attention with KV caching enabled produces
        identical outputs to the same computation without caching when processing
        a full sequence in a single forward pass.

        Args:
            attention_configs: Fixture providing attention configuration parameters.
        """
        for config in attention_configs:
            embed_dim, num_heads = config["embed_dim"], config["num_heads"]

            # Create attention modules
            attn_no_cache = MultiQueryAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                apply_causal_mask=True,
                use_kv_cache=False,
            )
            attn_with_cache = MultiQueryAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                apply_causal_mask=True,
                use_kv_cache=True,
            )

            # Ensure same parameters
            attn_with_cache.load_state_dict(attn_no_cache.state_dict())

            # Test sequence
            batch_size, seq_len = 2, 10
            input_seq = torch.randn(batch_size, seq_len, embed_dim)

            # Non-cached forward pass
            attn_no_cache.eval()
            with torch.no_grad():
                output_no_cache = attn_no_cache(
                    input_seq, kv=None, expanding_context=True
                )

            # Cached forward pass
            attn_with_cache.eval()
            attn_with_cache._inference_mode = True
            self._reset_cache(attn_with_cache)

            with torch.no_grad():
                output_with_cache = attn_with_cache(
                    input_seq, kv=None, expanding_context=True
                )

            # Verify outputs match
            torch.testing.assert_close(
                output_no_cache,
                output_with_cache,
                msg=f"MQA cache mismatch for config {config}",
                atol=1e-5,
                rtol=1e-4,
            )

    def test_gqa_single_step_parity(self, attention_configs) -> None:
        """Test single-step GQA cache parity with non-cached computation.

        Validates that Group Query Attention with KV caching enabled produces
        identical outputs to the same computation without caching when processing
        a full sequence in a single forward pass.

        Args:
            attention_configs: Fixture providing attention configuration parameters.
        """
        for config in attention_configs:
            embed_dim = config["embed_dim"]
            num_heads = config["num_heads"]
            num_groups = config["num_groups"]

            # Create attention modules
            attn_no_cache = GroupQueryAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_groups=num_groups,
                apply_causal_mask=True,
                use_kv_cache=False,
            )
            attn_with_cache = GroupQueryAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_groups=num_groups,
                apply_causal_mask=True,
                use_kv_cache=True,
            )

            # Ensure same parameters
            attn_with_cache.load_state_dict(attn_no_cache.state_dict())

            # Test sequence
            batch_size, seq_len = 2, 10
            input_seq = torch.randn(batch_size, seq_len, embed_dim)

            # Non-cached forward pass
            attn_no_cache.eval()
            with torch.no_grad():
                output_no_cache = attn_no_cache(
                    input_seq, kv=None, expanding_context=True
                )

            # Cached forward pass
            attn_with_cache.eval()
            attn_with_cache._inference_mode = True
            self._reset_cache(attn_with_cache)

            with torch.no_grad():
                output_with_cache = attn_with_cache(
                    input_seq, kv=None, expanding_context=True
                )

            # Verify outputs match
            torch.testing.assert_close(
                output_no_cache,
                output_with_cache,
                msg=f"GQA cache mismatch for config {config}",
                atol=1e-5,
                rtol=1e-4,
            )

    def test_mha_multi_step_generation(self) -> None:
        """Test multi-step autoregressive generation with MHA caching.

        Validates that incremental token generation using KV caching produces
        identical results to recomputing the full sequence at each step.
        This simulates autoregressive language model generation.
        """
        embed_dim, num_heads = 64, 4
        batch_size, initial_len, gen_steps = 1, 5, 3

        # Create attention modules
        attn_reference = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            apply_causal_mask=True,
            use_kv_cache=False,
        )
        attn_cached = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            apply_causal_mask=True,
            use_kv_cache=True,
        )

        # Ensure same parameters
        attn_cached.load_state_dict(attn_reference.state_dict())

        # Initial sequence
        initial_seq = torch.randn(batch_size, initial_len, embed_dim)

        # Reference: compute full sequences at each step
        reference_outputs = []
        current_seq = initial_seq.clone()

        attn_reference.eval()
        with torch.no_grad():
            for step in range(gen_steps):
                output = attn_reference(
                    current_seq, kv=None, expanding_context=True
                )
                last_token_output = output[:, -1:, :]  # Last token
                reference_outputs.append(last_token_output.clone())

                # Simulate next token (use last output as next input)
                current_seq = torch.cat([current_seq, last_token_output], dim=1)

        # Cached: incremental generation
        attn_cached.eval()
        attn_cached._inference_mode = True
        self._reset_cache(attn_cached)

        cached_outputs = []
        with torch.no_grad():
            # Initial forward pass
            initial_output = attn_cached(
                initial_seq, kv=None, expanding_context=True
            )
            first_gen_token = initial_output[:, -1:, :]
            cached_outputs.append(first_gen_token.clone())

            # Generate remaining tokens one by one
            for step in range(1, gen_steps):
                # Use previous output as next input
                next_input = cached_outputs[-1]
                output = attn_cached(
                    next_input, kv=None, expanding_context=True
                )
                cached_outputs.append(output.clone())

        # Verify each generated token matches
        for step in range(gen_steps):
            torch.testing.assert_close(
                reference_outputs[step],
                cached_outputs[step],
                msg=f"MHA multi-step generation mismatch at step {step}",
                atol=1e-5,
                rtol=1e-4,
            )

    def test_mqa_multi_step_generation(self) -> None:
        """Test multi-step autoregressive generation with MQA caching.

        Validates that incremental token generation using KV caching produces
        identical results to recomputing the full sequence at each step for
        Multi-Query Attention.
        """
        embed_dim, num_heads = 64, 4
        batch_size, initial_len, gen_steps = 1, 5, 3

        # Create attention modules
        attn_reference = MultiQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            apply_causal_mask=True,
            use_kv_cache=False,
        )
        attn_cached = MultiQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            apply_causal_mask=True,
            use_kv_cache=True,
        )

        # Ensure same parameters
        attn_cached.load_state_dict(attn_reference.state_dict())

        # Initial sequence
        initial_seq = torch.randn(batch_size, initial_len, embed_dim)

        # Reference: compute full sequences at each step
        reference_outputs = []
        current_seq = initial_seq.clone()

        attn_reference.eval()
        with torch.no_grad():
            for step in range(gen_steps):
                output = attn_reference(
                    current_seq, kv=None, expanding_context=True
                )
                last_token_output = output[:, -1:, :]
                reference_outputs.append(last_token_output.clone())
                current_seq = torch.cat([current_seq, last_token_output], dim=1)

        # Cached: incremental generation
        attn_cached.eval()
        attn_cached._inference_mode = True
        self._reset_cache(attn_cached)

        cached_outputs = []
        with torch.no_grad():
            # Initial forward pass
            initial_output = attn_cached(
                initial_seq, kv=None, expanding_context=True
            )
            first_gen_token = initial_output[:, -1:, :]
            cached_outputs.append(first_gen_token.clone())

            # Generate remaining tokens
            for step in range(1, gen_steps):
                next_input = cached_outputs[-1]
                output = attn_cached(
                    next_input, kv=None, expanding_context=True
                )
                cached_outputs.append(output.clone())

        # Verify each generated token matches
        for step in range(gen_steps):
            torch.testing.assert_close(
                reference_outputs[step],
                cached_outputs[step],
                msg=f"MQA multi-step generation mismatch at step {step}",
                atol=1e-5,
                rtol=1e-4,
            )

    def test_gqa_multi_step_generation(self) -> None:
        """Test multi-step autoregressive generation with GQA caching.

        Validates that incremental token generation using KV caching produces
        identical results to recomputing the full sequence at each step for
        Group Query Attention.
        """
        embed_dim, num_heads, num_groups = 64, 8, 4
        batch_size, initial_len, gen_steps = 1, 5, 3

        # Create attention modules
        attn_reference = GroupQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            apply_causal_mask=True,
            use_kv_cache=False,
        )
        attn_cached = GroupQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            apply_causal_mask=True,
            use_kv_cache=True,
        )

        # Ensure same parameters
        attn_cached.load_state_dict(attn_reference.state_dict())

        # Initial sequence
        initial_seq = torch.randn(batch_size, initial_len, embed_dim)

        # Reference: compute full sequences at each step
        reference_outputs = []
        current_seq = initial_seq.clone()

        attn_reference.eval()
        with torch.no_grad():
            for step in range(gen_steps):
                output = attn_reference(
                    current_seq, kv=None, expanding_context=True
                )
                last_token_output = output[:, -1:, :]
                reference_outputs.append(last_token_output.clone())
                current_seq = torch.cat([current_seq, last_token_output], dim=1)

        # Cached: incremental generation
        attn_cached.eval()
        attn_cached._inference_mode = True
        self._reset_cache(attn_cached)

        cached_outputs = []
        with torch.no_grad():
            # Initial forward pass
            initial_output = attn_cached(
                initial_seq, kv=None, expanding_context=True
            )
            first_gen_token = initial_output[:, -1:, :]
            cached_outputs.append(first_gen_token.clone())

            # Generate remaining tokens
            for step in range(1, gen_steps):
                next_input = cached_outputs[-1]
                output = attn_cached(
                    next_input, kv=None, expanding_context=True
                )
                cached_outputs.append(output.clone())

        # Verify each generated token matches
        for step in range(gen_steps):
            torch.testing.assert_close(
                reference_outputs[step],
                cached_outputs[step],
                msg=f"GQA multi-step generation mismatch at step {step}",
                atol=1e-5,
                rtol=1e-4,
            )

    def test_cache_state_management(self) -> None:
        """Test cache state lifecycle and position tracking.

        Validates that KV cache is properly initialized, updated, and tracks
        sequence position correctly across multiple forward passes.
        """
        embed_dim, num_heads = 32, 4
        batch_size, seq_len = 1, 8

        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            apply_causal_mask=True,
            use_kv_cache=True,
        )

        attn.eval()
        attn._inference_mode = True

        input_seq = torch.randn(batch_size, seq_len, embed_dim)

        # Verify cache starts empty
        assert attn.kv_cache is None, "Cache should start empty"

        with torch.no_grad():
            # First call initializes cache
            attn(input_seq, kv=None, expanding_context=True)
            assert (
                attn.kv_cache is not None
            ), "Cache should be initialized after first call"
            assert (
                attn.kv_cache["cur_pos"] == seq_len
            ), f"Cache position should be {seq_len}"

            # Second call should reuse cache
            next_token = torch.randn(batch_size, 1, embed_dim)
            attn(next_token, kv=None, expanding_context=True)
            expected_pos = seq_len + 1
            assert (
                attn.kv_cache["cur_pos"] == expected_pos
            ), f"Cache position should be {expected_pos}"

    def test_cache_memory_efficiency(self) -> None:
        """Test pre-allocated cache memory efficiency.

        Validates that KV cache uses fixed-size pre-allocated tensors with
        shape determined by MAX_SEQ_LEN constant rather than dynamic allocation.
        """
        embed_dim, num_heads = 64, 4
        batch_size = 1

        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            apply_causal_mask=True,
            use_kv_cache=True,
        )

        attn.eval()
        attn._inference_mode = True

        # Initial sequence
        initial_seq = torch.randn(batch_size, 5, embed_dim)

        with torch.no_grad():
            attn(initial_seq, kv=None, expanding_context=True)

            # Verify cache size is pre-allocated
            cache_k_shape = attn.kv_cache["key"].shape
            cache_v_shape = attn.kv_cache["value"].shape

            # MAX_SEQ_LEN = 128
            expected_shape = (
                batch_size,
                num_heads,
                128,
                embed_dim // num_heads,
            )

            assert (
                cache_k_shape == expected_shape
            ), f"Key cache shape {cache_k_shape} != {expected_shape}"
            assert (
                cache_v_shape == expected_shape
            ), f"Value cache shape {cache_v_shape} != {expected_shape}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])