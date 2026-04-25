"""
State encoder for the policy network.
MLP-based (replaces GNN from v3 design — too complex for hackathon timeline).
Document: GNN would be used in production for the delegation graph component.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    Encodes the flat state vector into a compressed representation.
    The SB3 policy will use this as its feature extractor.

    Architecture:
      - Input: flat state vector (~1376 + N*768 dims)
      - Hidden: 512 → 256 → 128
      - Output: 128-dim feature vector

    Note: The MLP operates on the full flat vector including:
      - Task embedding (384)
      - Roster + called specialist embeddings (padded)
      - Graph adjacency vector (100)
      - Scratchpad summary (384)
      - Scalar features (8)
    This is the "MLP adjacency" approach that replaces the GNN.
    """

    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
