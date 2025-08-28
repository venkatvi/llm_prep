"""
Speculative Decoding Implementation.

This module implements speculative decoding, a technique for accelerating autoregressive
text generation by using a smaller "draft" model to propose tokens and a larger "target"
model to verify them in parallel.

The implementation includes:
- SpecDecodingConfig: Configuration for draft and target models
- SpecDecodingPair: Main class implementing the speculative decoding algorithm
- Acceptance/rejection sampling with corrected distribution resampling
- Token-to-embedding conversion utilities

Copyright (c) 2025. All rights reserved.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# flake8: noqa: E402
from lib.utils import set_seed
from regression.configs import TransformerModelConfig
from regression.h_transformer import ARTransformerModel

set_seed(42)

"""
Spec Decoding Example: 

Model Configuration:
  - Layer ratio: 2 vs 8 layers (4x difference) ✅
  - Size scaling: 64→1024 embed_dim, 128→4096 ffn_latent_dim ✅
  - Draft MOE: Disabled for speed ✅
  - Attention progression: GQA→MHA for quality ✅
  - expanding_context: True for draft, False for target ✅

Implementation Pattern:

  - Sequential draft generation: Perfect for cache building ✅
  - Token collection: Collecting generated tokens correctly ✅

"""


@dataclass
class SpecDecodingConfig:
    """Configuration for speculative decoding with draft and target models.

    Attributes:
        draft_config: Configuration for the fast draft model
        target_config: Configuration for the high-quality target model
        draft_steps: Number of tokens to generate with draft model before verification
    """

    draft_config: TransformerModelConfig
    target_config: TransformerModelConfig
    draft_steps: int


class SpecDecodingPair(torch.nn.Module):
    """Speculative decoding implementation using draft and target transformer models.

    This class implements the core speculative decoding algorithm:
    1. Generate k tokens with fast draft model (sequential)
    2. Verify all k tokens with target model (parallel)
    3. Accept/reject tokens based on probability ratios
    4. Resample rejected tokens from corrected distribution

    Attributes:
        config: Speculative decoding configuration
        draft_model: Fast model for token proposal
        target_model: High-quality model for verification
    """

    def __init__(self, spec_decoding_config: SpecDecodingConfig) -> None:
        """Initialize speculative decoding pair.

        Args:
            spec_decoding_config: Configuration containing draft and target model configs
        """
        super().__init__()
        self.config = spec_decoding_config
        self.draft_model = ARTransformerModel(self.config.draft_config)
        self.target_model = ARTransformerModel(self.config.target_config)

    def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Generate n tokens using speculative decoding.

        Args:
            x: Input sequence tensor of shape [batch_size, seq_len, input_dim]
            n: Number of tokens to generate

        Returns:
            Extended sequence tensor of shape [batch_size, seq_len + n, input_dim]
        """
        current_output = x.clone()
        iter_count = 0

        while (current_output.size(1) - x.size(1)) < n:
            print(f"Speculative decoding iteration: {iter_count}")

            # Phase 1: Generate draft tokens
            print(f"Generating {self.config.draft_steps} draft tokens")
            draft_sequence, draft_probs = self.generate_draft_sequence(current_output)

            # Phase 2: Verify with target model
            print("Verifying draft tokens with target model")
            full_sequence = torch.cat([current_output, draft_sequence], dim=1)
            target_probs = self.verify_draft_with_target(full_sequence)

            # Phase 3: Accept/reject tokens
            print("Performing acceptance/rejection sampling")
            accepted_count = self.accept_or_reject(draft_probs, target_probs)

            # Phase 4: Resample rejected tokens
            print("Resampling rejected tokens")
            final_count, resampled_embeddings = self.resample(
                draft_probs, target_probs, accepted_count
            )

            # Assemble final sequence
            if final_count <= 0:
                print("No tokens accepted: falling back to target model")
                next_token = self.target_model.generate_next_token(
                    current_output, expanding_context=True
                )
                current_output = torch.cat([current_output, next_token], dim=1)
            else:
                print(f"Accepted {final_count} tokens from draft/resampling")
                if accepted_count > 0:
                    accepted_sequence = draft_sequence[:, :accepted_count, :]
                else:
                    accepted_sequence = torch.empty(
                        draft_sequence.size(0), 0, draft_sequence.size(2)
                    )

                if resampled_embeddings:
                    resampled_sequence = torch.cat(resampled_embeddings, dim=1)
                    current_output = torch.cat(
                        [current_output, accepted_sequence, resampled_sequence], dim=1
                    )
                else:
                    current_output = torch.cat(
                        [current_output, accepted_sequence], dim=1
                    )

            iter_count += 1

        return current_output

    def generate_draft_sequence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft token sequence using the draft model.

        Args:
            x: Input sequence tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Tuple of:
            - draft_sequence: Generated embeddings [batch_size, draft_steps, output_dim]
            - draft_probs: Token probabilities [batch_size, draft_steps, vocab_size]
        """
        generated_draft_embeddings = []
        generated_draft_logits = []
        current_sequence = x.clone()  # [bs, seq_len, input_dim]

        for step in range(self.config.draft_steps):
            print(f"Draft generation step {step+1}/{self.config.draft_steps}")
            # Generate next token embedding and logits
            embedding, logits = (
                self.draft_model.generate_next_token_embedding_and_logits(
                    current_sequence, expanding_context=True
                )
            )
            generated_draft_embeddings.append(embedding)  # [bs, 1, output_dim]
            generated_draft_logits.append(logits)  # [bs, 1, vocab_size]
            current_sequence = torch.cat([current_sequence, embedding], dim=1)

        draft_sequence = torch.cat(
            generated_draft_embeddings, dim=1
        )  # [bs, draft_steps, output_dim]
        draft_logits = torch.cat(
            generated_draft_logits, dim=1
        )  # [bs, draft_steps, vocab_size]
        draft_probs = torch.softmax(
            draft_logits, dim=-1
        )  # [bs, draft_steps, vocab_size]
        return draft_sequence, draft_probs

    def verify_draft_with_target(self, x: torch.Tensor) -> torch.Tensor:
        """Verify draft tokens using the target model in parallel.

        Args:
            x: Full sequence including draft tokens [batch_size, seq_len + draft_steps, input_dim]

        Returns:
            Target model probabilities for draft tokens [batch_size, draft_steps, vocab_size]
        """
        target_logits = self.target_model.get_logits(x, expanding_context=False)
        return torch.softmax(target_logits[:, -self.config.draft_steps :, :], dim=-1)

    def accept_or_reject(
        self, draft_probs: torch.Tensor, target_probs: torch.Tensor
    ) -> int:
        """Accept or reject draft tokens based on probability ratios.

        Uses rejection sampling with acceptance probability min(1, p_target/p_draft).
        Stops at first rejection to maintain sequence coherence.

        Args:
            draft_probs: Draft model probabilities [batch_size, draft_steps, vocab_size]
            target_probs: Target model probabilities [batch_size, draft_steps, vocab_size]

        Returns:
            Number of accepted tokens (0 to draft_steps)
        """
        accepted_count = 0
        for k in range(draft_probs.size(1)):
            # Sample token from draft distribution
            sampled_draft_token_id = torch.multinomial(
                draft_probs[:, k, :], num_samples=1
            )
            pD = draft_probs[:, k, :].gather(1, sampled_draft_token_id)  # [bs, 1]
            pT = target_probs[:, k, :].gather(1, sampled_draft_token_id)  # [bs, 1]

            # Compute acceptance ratio: min(1, p_target/p_draft)
            acceptance_ratio = (pT / pD).clamp(max=1.0)  # [bs, 1]

            # Accept if all samples in batch pass Bernoulli test
            if torch.bernoulli(acceptance_ratio).all():
                accepted_count += 1
            else:
                break  # Stop at first rejection
        return accepted_count

    def resample(
        self, draft_probs: torch.Tensor, target_probs: torch.Tensor, accepted_count: int
    ) -> Tuple[int, List[torch.Tensor]]:
        """Resample rejected tokens using corrected distribution.

        For tokens that were rejected, sample from the corrected distribution
        max(0, p_target - p_draft) normalized to a valid probability distribution.

        Args:
            draft_probs: Draft model probabilities [batch_size, draft_steps, vocab_size]
            target_probs: Target model probabilities [batch_size, draft_steps, vocab_size]
            accepted_count: Number of tokens already accepted

        Returns:
            Tuple of:
            - final_count: Total number of tokens (accepted + resampled)
            - resampled_embeddings: List of resampled token embeddings
        """
        final_count = accepted_count
        resampled_embeddings = []

        # Resample all remaining positions [accepted_count:draft_steps]
        for k in range(accepted_count, self.config.draft_steps):
            pD = draft_probs[:, k, :]  # [bs, vocab_size]
            pT = target_probs[:, k, :]  # [bs, vocab_size]

            # Corrected distribution: max(0, p_target - p_draft)
            corrected_probs = torch.clamp(pT - pD, min=0.0)  # [bs, vocab_size]

            # Normalize probability distribution
            corrected_probs_sum = corrected_probs.sum(dim=-1, keepdim=True)  # [bs, 1]

            # Handle edge case where all corrected probs are zero
            corrected_probs_sum = torch.where(
                corrected_probs_sum == 0,
                torch.ones_like(corrected_probs_sum),
                corrected_probs_sum,
            )
            corrected_probs = corrected_probs / corrected_probs_sum  # [bs, vocab_size]

            # Sample from corrected distribution
            resampled_token_id = torch.multinomial(
                corrected_probs, num_samples=1
            )  # [bs, 1]

            # Convert token_id back to embedding
            resampled_embedding = self.token_to_embeddings(
                resampled_token_id
            )  # [bs, 1, output_dim]
            resampled_embeddings.append(resampled_embedding)
            final_count += 1

        return final_count, resampled_embeddings

    def token_to_embeddings(self, token_id: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings using lm_head weight approximation.

        Uses the transpose of the language model head weights to approximate
        the embedding lookup. This is a simplified approach - in production,
        you would use a proper token embedding table.

        Args:
            token_id: Token IDs tensor of shape [batch_size, 1]

        Returns:
            Token embeddings tensor of shape [batch_size, 1, output_dim]
        """
        lm_head_weight = (
            self.target_model.model.lm_head.weight
        )  # [vocab_size, output_dim]

        # Create one-hot encoding
        one_hot = torch.nn.functional.one_hot(
            token_id, num_classes=lm_head_weight.size(0)
        ).float()  # [bs, 1, vocab_size]

        # Matrix multiply to get embedding approximation
        embedding = torch.matmul(one_hot, lm_head_weight)  # [bs, 1, output_dim]
        return embedding
