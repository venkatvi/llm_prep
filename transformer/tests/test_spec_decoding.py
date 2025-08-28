"""
Test suite for speculative decoding implementation.

Tests the SpecDecodingPair class and related components including:
- Configuration validation
- Draft and target model initialization
- Token generation and acceptance logic
- Resampling and embedding conversion
- Integration with different attention mechanisms

Copyright (c) 2025. All rights reserved.
"""

import os
import sys
import unittest
from typing import Tuple

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from regression.configs import TransformerModelConfig, AutoregressiveDecodeConfig
from transformer.configs import FFNConfig
from transformer.spec_decoding import SpecDecodingPair, SpecDecodingConfig


class TestSpecDecodingConfig(unittest.TestCase):
    """Test SpecDecodingConfig dataclass."""

    def setUp(self):
        """Set up test configurations."""
        self.draft_ffn_config = FFNConfig(
            embed_dim=64, latent_dim=128, use_moe=False
        )
        
        self.target_ffn_config = FFNConfig(
            embed_dim=256, latent_dim=512, use_moe=False
        )

        self.draft_config = TransformerModelConfig(
            name="draft",
            max_seq_len=128,
            input_dim=1,
            embed_dim=64,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=2,
            output_dim=1,
            apply_causal_mask=True,
            autoregressive_mode=True,
            decode_config=AutoregressiveDecodeConfig(
                num_steps=10,
                expanding_context=True,
                max_seq_len=128,
                use_kv_cache=True,
            ),
            attention_type="mqa",
            ffn_config=self.draft_ffn_config,
            vocab_size=100,
        )

        self.target_config = TransformerModelConfig(
            name="target",
            max_seq_len=128,
            input_dim=1,
            embed_dim=256,
            ffn_latent_dim=512,
            num_layers=4,
            num_heads=8,
            num_groups=4,
            output_dim=1,
            apply_causal_mask=True,
            autoregressive_mode=True,
            decode_config=AutoregressiveDecodeConfig(
                num_steps=10,
                expanding_context=False,
                max_seq_len=128,
                use_kv_cache=False,
            ),
            attention_type="mha",
            ffn_config=self.target_ffn_config,
            vocab_size=100,
        )

    def test_spec_decoding_config_creation(self):
        """Test SpecDecodingConfig creation."""
        config = SpecDecodingConfig(
            draft_config=self.draft_config,
            target_config=self.target_config,
            draft_steps=5,
        )
        
        self.assertEqual(config.draft_steps, 5)
        self.assertEqual(config.draft_config.name, "draft")
        self.assertEqual(config.target_config.name, "target")
        self.assertEqual(config.draft_config.num_layers, 2)
        self.assertEqual(config.target_config.num_layers, 4)


class TestSpecDecodingPair(unittest.TestCase):
    """Test SpecDecodingPair class."""

    def setUp(self):
        """Set up test configurations and models."""
        # Small configurations for fast testing
        self.draft_ffn_config = FFNConfig(
            embed_dim=32, latent_dim=64, use_moe=False
        )
        
        self.target_ffn_config = FFNConfig(
            embed_dim=64, latent_dim=128, use_moe=False
        )

        self.draft_config = TransformerModelConfig(
            name="test_draft",
            max_seq_len=64,
            input_dim=1,
            embed_dim=32,
            ffn_latent_dim=64,
            num_layers=1,
            num_heads=2,
            num_groups=1,
            output_dim=1,
            apply_causal_mask=True,
            autoregressive_mode=True,
            decode_config=AutoregressiveDecodeConfig(
                num_steps=5,
                expanding_context=True,
                max_seq_len=64,
                use_kv_cache=True,
            ),
            attention_type="mqa",
            ffn_config=self.draft_ffn_config,
            vocab_size=50,
        )

        self.target_config = TransformerModelConfig(
            name="test_target",
            max_seq_len=64,
            input_dim=1,
            embed_dim=64,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=2,
            output_dim=1,
            apply_causal_mask=True,
            autoregressive_mode=True,
            decode_config=AutoregressiveDecodeConfig(
                num_steps=5,
                expanding_context=False,
                max_seq_len=64,
                use_kv_cache=False,
            ),
            attention_type="mha",
            ffn_config=self.target_ffn_config,
            vocab_size=50,
        )

        self.spec_config = SpecDecodingConfig(
            draft_config=self.draft_config,
            target_config=self.target_config,
            draft_steps=3,
        )

    def test_model_initialization(self):
        """Test SpecDecodingPair initialization."""
        model = SpecDecodingPair(self.spec_config)
        
        self.assertIsNotNone(model.draft_model)
        self.assertIsNotNone(model.target_model)
        self.assertEqual(model.config.draft_steps, 3)

    def test_generate_draft_sequence(self):
        """Test draft sequence generation."""
        model = SpecDecodingPair(self.spec_config)
        
        # Test input
        batch_size, seq_len, input_dim = 2, 4, 1
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Generate draft sequence
        draft_sequence, draft_probs = model.generate_draft_sequence(x)
        
        # Check shapes
        expected_draft_shape = (batch_size, self.spec_config.draft_steps, 1)
        expected_probs_shape = (batch_size, self.spec_config.draft_steps, 50)
        
        self.assertEqual(draft_sequence.shape, expected_draft_shape)
        self.assertEqual(draft_probs.shape, expected_probs_shape)
        
        # Check that probabilities sum to 1
        prob_sums = draft_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6))

    def test_verify_draft_with_target(self):
        """Test target model verification."""
        model = SpecDecodingPair(self.spec_config)
        
        # Create full sequence (input + draft)
        batch_size, seq_len, input_dim = 2, 4, 1
        full_seq_len = seq_len + self.spec_config.draft_steps
        x = torch.randn(batch_size, full_seq_len, input_dim)
        
        # Verify with target
        target_probs = model.verify_draft_with_target(x)
        
        # Check shape
        expected_shape = (batch_size, self.spec_config.draft_steps, 50)
        self.assertEqual(target_probs.shape, expected_shape)
        
        # Check that probabilities sum to 1
        prob_sums = target_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6))

    def test_accept_or_reject(self):
        """Test acceptance/rejection logic."""
        model = SpecDecodingPair(self.spec_config)
        
        # Create mock probability distributions
        batch_size = 2
        draft_steps = self.spec_config.draft_steps
        vocab_size = 50
        
        draft_probs = torch.softmax(torch.randn(batch_size, draft_steps, vocab_size), dim=-1)
        target_probs = torch.softmax(torch.randn(batch_size, draft_steps, vocab_size), dim=-1)
        
        # Test acceptance
        accepted_count = model.accept_or_reject(draft_probs, target_probs)
        
        # Should return integer between 0 and draft_steps
        self.assertIsInstance(accepted_count, int)
        self.assertGreaterEqual(accepted_count, 0)
        self.assertLessEqual(accepted_count, draft_steps)

    def test_resample(self):
        """Test resampling logic."""
        model = SpecDecodingPair(self.spec_config)
        
        # Create mock probability distributions
        batch_size = 2
        draft_steps = self.spec_config.draft_steps
        vocab_size = 50
        
        draft_probs = torch.softmax(torch.randn(batch_size, draft_steps, vocab_size), dim=-1)
        target_probs = torch.softmax(torch.randn(batch_size, draft_steps, vocab_size), dim=-1)
        
        # Test resampling with partial acceptance
        accepted_count = 1
        final_count, resampled_embeddings = model.resample(
            draft_probs, target_probs, accepted_count
        )
        
        # Check results
        expected_resampled = draft_steps - accepted_count
        self.assertEqual(final_count, draft_steps)
        self.assertEqual(len(resampled_embeddings), expected_resampled)
        
        if resampled_embeddings:
            # Check embedding shape
            embedding_shape = resampled_embeddings[0].shape
            expected_embedding_shape = (batch_size, 1, 1)  # [batch, 1, output_dim]
            self.assertEqual(embedding_shape, expected_embedding_shape)

    def test_token_to_embeddings(self):
        """Test token to embedding conversion."""
        model = SpecDecodingPair(self.spec_config)
        
        # Test token conversion
        batch_size = 2
        token_ids = torch.randint(0, 50, (batch_size, 1))
        
        embeddings = model.token_to_embeddings(token_ids)
        
        # Check shape
        expected_shape = (batch_size, 1, 1)  # [batch, 1, output_dim]
        self.assertEqual(embeddings.shape, expected_shape)

    def test_forward_pass_basic(self):
        """Test basic forward pass with small generation."""
        model = SpecDecodingPair(self.spec_config)
        
        # Test input
        batch_size, seq_len, input_dim = 1, 3, 1
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Generate small number of tokens
        n_tokens = 2
        output = model(x, n_tokens)
        
        # Check output shape - the actual generation might produce more tokens than requested
        # due to speculative decoding iterations
        self.assertEqual(output.size(0), batch_size)  # Batch size should match
        self.assertEqual(output.size(2), input_dim)   # Input dim should match
        self.assertGreaterEqual(output.size(1), seq_len + n_tokens)  # At least the requested tokens

    def test_edge_case_no_tokens_accepted(self):
        """Test edge case where no tokens are accepted."""
        model = SpecDecodingPair(self.spec_config)
        
        # Create biased probabilities that will likely be rejected
        batch_size = 1
        draft_steps = self.spec_config.draft_steps
        vocab_size = 50
        
        # Draft model assigns high probability to first token
        draft_probs = torch.zeros(batch_size, draft_steps, vocab_size)
        draft_probs[:, :, 0] = 1.0
        
        # Target model assigns high probability to last token  
        target_probs = torch.zeros(batch_size, draft_steps, vocab_size)
        target_probs[:, :, -1] = 1.0
        
        # This should result in very low acceptance
        accepted_count = model.accept_or_reject(draft_probs, target_probs)
        
        # Test resampling handles zero acceptance
        final_count, resampled_embeddings = model.resample(
            draft_probs, target_probs, accepted_count
        )
        
        # Should resample all positions
        self.assertEqual(len(resampled_embeddings), draft_steps - accepted_count)

    def test_different_attention_types(self):
        """Test speculative decoding with different attention mechanisms."""
        attention_types = ["mha", "mqa", "gqa"]
        
        for draft_attn in attention_types:
            for target_attn in attention_types:
                with self.subTest(draft=draft_attn, target=target_attn):
                    # Update configs
                    self.draft_config.attention_type = draft_attn
                    self.target_config.attention_type = target_attn
                    
                    spec_config = SpecDecodingConfig(
                        draft_config=self.draft_config,
                        target_config=self.target_config,
                        draft_steps=2,
                    )
                    
                    # Should initialize without error
                    model = SpecDecodingPair(spec_config)
                    self.assertIsNotNone(model.draft_model)
                    self.assertIsNotNone(model.target_model)

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        model = SpecDecodingPair(self.spec_config)
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            with self.subTest(batch_size=batch_size):
                seq_len, input_dim = 3, 1
                x = torch.randn(batch_size, seq_len, input_dim)
                
                # Test draft generation
                draft_sequence, draft_probs = model.generate_draft_sequence(x)
                
                expected_draft_shape = (batch_size, self.spec_config.draft_steps, 1)
                expected_probs_shape = (batch_size, self.spec_config.draft_steps, 50)
                
                self.assertEqual(draft_sequence.shape, expected_draft_shape)
                self.assertEqual(draft_probs.shape, expected_probs_shape)


class TestSpecDecodingIntegration(unittest.TestCase):
    """Integration tests for speculative decoding."""

    def setUp(self):
        """Set up test configurations with MOE support."""
        self.draft_ffn_config = FFNConfig(
            embed_dim=32, latent_dim=64, use_moe=False
        )
        
        self.target_moe_ffn_config = FFNConfig(
            embed_dim=64,
            latent_dim=128,
            use_moe=True,
            num_experts=4,
            capacity=16,
            alpha=0.01,
            topk=1,
        )

        self.draft_config = TransformerModelConfig(
            name="test_draft_moe",
            max_seq_len=32,
            input_dim=1,
            embed_dim=32,
            ffn_latent_dim=64,
            num_layers=1,
            num_heads=2,
            num_groups=1,
            output_dim=1,
            apply_causal_mask=True,
            autoregressive_mode=True,
            decode_config=AutoregressiveDecodeConfig(
                num_steps=3,
                expanding_context=True,
                max_seq_len=32,
                use_kv_cache=True,
            ),
            attention_type="mqa",
            ffn_config=self.draft_ffn_config,
            vocab_size=20,
        )

        self.target_config = TransformerModelConfig(
            name="test_target_moe",
            max_seq_len=32,
            input_dim=1,
            embed_dim=64,
            ffn_latent_dim=128,
            num_layers=2,
            num_heads=4,
            num_groups=2,
            output_dim=1,
            apply_causal_mask=True,
            autoregressive_mode=True,
            decode_config=AutoregressiveDecodeConfig(
                num_steps=3,
                expanding_context=False,
                max_seq_len=32,
                use_kv_cache=False,
            ),
            attention_type="mha",
            ffn_config=self.target_moe_ffn_config,
            vocab_size=20,
        )

    def test_speculative_decoding_with_moe(self):
        """Test speculative decoding with MOE in target model."""
        spec_config = SpecDecodingConfig(
            draft_config=self.draft_config,
            target_config=self.target_config,
            draft_steps=2,
        )
        
        model = SpecDecodingPair(spec_config)
        
        # Test generation
        batch_size, seq_len, input_dim = 1, 2, 1
        x = torch.randn(batch_size, seq_len, input_dim)
        
        n_tokens = 3
        output = model(x, n_tokens)
        
        # Check output shape - the actual generation might produce more tokens than requested
        # due to speculative decoding iterations
        self.assertEqual(output.size(0), batch_size)  # Batch size should match
        self.assertEqual(output.size(2), input_dim)   # Input dim should match
        self.assertGreaterEqual(output.size(1), seq_len + n_tokens)  # At least the requested tokens

    def test_end_to_end_generation(self):
        """Test end-to-end token generation."""
        spec_config = SpecDecodingConfig(
            draft_config=self.draft_config,
            target_config=self.target_config,
            draft_steps=2,
        )
        
        model = SpecDecodingPair(spec_config)
        
        # Set eval mode for deterministic behavior
        model.eval()
        
        with torch.no_grad():
            # Test generation
            batch_size, seq_len, input_dim = 2, 2, 1
            x = torch.randn(batch_size, seq_len, input_dim)
            
            # Generate tokens
            n_tokens = 4
            output = model(x, n_tokens)
            
            # Verify output shape and content
            expected_shape = (batch_size, seq_len + n_tokens, input_dim)
            self.assertEqual(output.shape, expected_shape)
            
            # Verify that original input is preserved
            self.assertTrue(torch.allclose(output[:, :seq_len, :], x, atol=1e-6))


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)