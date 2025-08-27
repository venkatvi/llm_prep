"""Mixture of Experts (MOE) implementation with capacity-based routing.

This module implements a sparse MOE layer that routes tokens to different expert networks
based on learned routing probabilities. It includes capacity constraints and overflow handling.
"""

import os
import sys
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from transformer.ffn import FFN


class MOE(torch.nn.Module):
    """Mixture of Experts layer with capacity-constrained routing.

    This implementation uses a learned router to assign input tokens to different
    expert networks. When an expert's capacity is exceeded, overflow tokens are
    handled by a dedicated overflow expert.

    Args:
        embed_dim (int): Input and output embedding dimension
        ffn_latent_dim (int): Hidden dimension for expert FFN layers
        num_experts (int): Number of expert networks
        capacity (float): Maximum number of tokens each expert can process
        alpha (float): Load balancing coefficient for auxiliary loss
        topk (int): Number of top experts to consider for routing
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_experts: int,
        capacity: float,
        alpha: float,
        topk: int,
    ):
        super().__init__()

        # Router network that assigns tokens to experts
        self.router = torch.nn.Linear(embed_dim, num_experts)

        # Expert networks
        self.experts = torch.nn.ModuleList(
            [
                FFN(embed_dim=embed_dim, latent_dim=ffn_latent_dim)
                for _ in range(num_experts)
            ]
        )

        # Overflow expert for handling capacity constraints
        self.overflow_expert = FFN(embed_dim=embed_dim, latent_dim=ffn_latent_dim)

        # Configuration parameters
        self.capacity = int(capacity)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.topk = topk
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with top-k expert routing and load balancing.

        Args:
            input (torch.Tensor): Input tensor of shape (B, S, D)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and auxiliary loss
        """
        B, S, D = input.size()

        # Step 1: Router computes expert assignment probabilities
        logits = self.router(input)  # Shape: (B, S, num_experts)
        scores = F.softmax(logits, dim=-1)  # Shape: (B, S, num_experts)

        # Step 2: Select top-k experts per token
        top_k_weights, expert_indices = torch.topk(
            scores, k=self.topk, dim=-1
        )  # Shape: (B, S, topk)

        # Step 3: Normalize weights among selected experts
        top_k_weights = top_k_weights / top_k_weights.sum(
            dim=-1, keepdim=True
        )  # Shape: (B, S, topk)

        # Step 4: Flatten tensors for expert processing
        flat_input = input.reshape([B * S, D])  # Shape: (B*S, D)
        topk_indices = expert_indices.reshape([B * S, self.topk])  # (B*S, topk)
        topk_weights = top_k_weights.reshape([B * S, self.topk])  # (B*S, topk)

        # Step 5: Initialize output and auxiliary loss
        out = torch.zeros_like(input, device=input.device).reshape([B * S, D])
        aux_loss = 0.0

        # Step 6: Process each expert
        for idx, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            mask_per_token = topk_indices == idx  # Shape: (B*S, topk)
            mask = mask_per_token.any(dim=-1)  # Shape: (B*S,)
            assigned_indices = mask.nonzero().squeeze(-1)  # Token indices

            # Skip if no tokens assigned to this expert
            if assigned_indices.numel() == 0:
                continue

            # Get inputs and weights for this expert
            input_per_expert = flat_input[assigned_indices]  # (num_assigned, D)
            w_expert = topk_weights * mask_per_token  # (B*S, topk)
            w_expert = w_expert.sum(dim=-1).unsqueeze(-1)  # (B*S, 1)
            w_expert_per_token = w_expert[assigned_indices]  # (num_assigned, 1)

            # Calculate load balancing metrics
            num_assigned = input_per_expert.size(0)
            f_expert = min(self.capacity, num_assigned) / flat_input.size(0)

            # Route tokens based on capacity
            if num_assigned <= self.capacity:
                # Expert can handle all assigned tokens
                expert_output = expert(input_per_expert)
                out[assigned_indices, :] += w_expert_per_token * expert_output
                p_expert = w_expert_per_token.mean(dim=0).item()
            else:
                # Handle capacity overflow
                input_within_cap = input_per_expert[: self.capacity, :]
                input_overflow = input_per_expert[self.capacity :, :]
                weight_within_cap = w_expert_per_token[: self.capacity, :]
                weight_overflow = w_expert_per_token[self.capacity :, :]

                # Process tokens within capacity with main expert
                if len(input_within_cap) > 0:
                    expert_output = expert(input_within_cap)
                    out[assigned_indices[: self.capacity], :] += (
                        weight_within_cap * expert_output
                    )
                    p_expert = weight_within_cap.mean(dim=0).item()
                else:
                    # No tokens within capacity, use overall average
                    p_expert = w_expert_per_token.mean(dim=0).item()

                # Process overflow tokens with overflow expert
                if len(input_overflow) > 0:
                    overflow_output = self.overflow_expert(input_overflow)
                    out[assigned_indices[self.capacity :], :] += (
                        weight_overflow * overflow_output
                    )

            # Accumulate load balancing loss
            aux_loss += f_expert * p_expert

        # Scale auxiliary loss
        aux_loss = self.alpha * self.num_experts * aux_loss

        return out.reshape([B, S, D]), aux_loss

    def forward_top1(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MOE layer.

        Routes input tokens to experts based on learned routing probabilities.
        When an expert exceeds capacity, overflow tokens are handled by the
        overflow expert.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, S, D = input.size()

        # Router computation
        router_out = self.router(input)  # Shape: (B, S, num_experts)
        logits = torch.softmax(router_out, dim=-1)  # Shape: (B, S, num_experts)

        # Expert assignment (currently top-1, topk parameter unused)
        expert_ids = torch.argmax(logits, dim=2)  # Shape: (B, S)

        # Flatten for easier processing
        flat_input = input.reshape(B * S, D)  # Shape: (B*S, D)
        flat_expert_ids = expert_ids.reshape(B * S)  # Shape: (B*S,)

        # Create expert assignment mask
        expert_mask = F.one_hot(
            flat_expert_ids, num_classes=self.num_experts
        ).float()  # Shape: (B*S, num_experts)

        # Initialize output with proper device placement
        out = torch.zeros([B * S, D], device=input.device)

        # Process each expert
        for idx, expert in enumerate(self.experts):
            current_expert_mask = expert_mask[:, idx]  # Shape: (B*S,)
            expert_tokens = flat_input * current_expert_mask.unsqueeze(
                -1
            )  # Shape: (B*S, D)

            num_assigned = current_expert_mask.sum().int()

            if num_assigned <= self.capacity:
                # Expert can handle all assigned tokens
                expert_output = expert(expert_tokens)
                out += expert_output * current_expert_mask.unsqueeze(-1)
            else:
                # Handle capacity overflow
                all_indices = current_expert_mask.nonzero().squeeze(-1)
                keep_indices = all_indices[: self.capacity]
                overflow_indices = all_indices[self.capacity :]

                # Process tokens within capacity
                if len(keep_indices) > 0:
                    expert_output = expert(flat_input[keep_indices])
                    out[keep_indices, :] += expert_output

                # Process overflow tokens
                if len(overflow_indices) > 0:
                    expert_output = self.overflow_expert(flat_input[overflow_indices])
                    out[overflow_indices, :] += expert_output

        return out.reshape([B, S, D])
