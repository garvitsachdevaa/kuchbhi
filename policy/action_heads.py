"""
Factored action heads for the policy.
4 heads decoded sequentially — avoids combinatorial explosion.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class FactoredActionHead(nn.Module):
    """
    4-head factored action network.
    In SB3, this is the 'pi' network (actor).
    """

    def __init__(
        self,
        input_dim: int,
        num_meta_actions: int = 8,
        num_delegation_modes: int = 7,
        max_specialists: int = 8,
        num_mode_params: int = 4,
    ):
        super().__init__()
        self.max_specialists = max_specialists

        # Head 1: Meta-action
        self.meta_head = nn.Linear(input_dim, num_meta_actions)

        # Head 2: Specialist selection (multi-label)
        self.specialist_head = nn.Linear(input_dim, max_specialists)

        # Head 3: Delegation mode
        self.mode_head = nn.Linear(input_dim, num_delegation_modes)

        # Head 4: Mode parameters (continuous)
        self.params_head = nn.Linear(input_dim, num_mode_params)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns flat action vector.
        Shape: (batch, 1 + max_specialists + 1 + num_mode_params)
        """
        meta = self.meta_head(features).argmax(dim=-1, keepdim=True).float()
        specialists = torch.sigmoid(self.specialist_head(features)) * 2 - 1
        mode = self.mode_head(features).argmax(dim=-1, keepdim=True).float()
        params = torch.tanh(self.params_head(features))
        return torch.cat([meta, specialists, mode, params], dim=-1)
