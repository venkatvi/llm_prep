"""Mixture of Experts (MOE) implementation with capacity-based routing.

This module implements a sparse MOE layer that routes tokens to different expert networks
based on learned routing probabilities. It includes capacity constraints and overflow handling.
"""

import os
import sys

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
        alpha (float): Load balancing coefficient (currently unused)
        topk (int): Number of top experts to consider (currently unused, defaults to top-1)
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
        self.alpha = alpha
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.topk = topk

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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

        # Initialize output
        out = torch.zeros([B * S, D])

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
