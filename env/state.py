"""
State Representation — Fully observable episode state for the RL policy.

State components:
  1. Task embedding (384-dim) — what needs to be done
  2. Roster embedding matrix (N × 384) — available specialists
  3. Called specialist embeddings (K × 384) — who has been called
  4. Delegation graph adjacency vector (100-dim) — call structure
  5. Scratchpad summary embedding (384-dim) — context so far
  6. Scalar features (8-dim) — step count, depth, costs, etc.
  7. Called specialist mask (N-dim) — binary, who's been called

Flattened total: ~1376 + N*384 dims (variable; padded to max_specialists)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EpisodeState:
    """
    Complete state for one timestep in an episode.
    Built by the SpindleFlowEnv at each step.
    """
    # Core semantic representations
    task_embedding: np.ndarray          # (384,)
    roster_embeddings: np.ndarray       # (max_specialists, 384)
    called_embeddings: np.ndarray       # (max_specialists, 384) — 0s for uncalled
    scratchpad_embedding: np.ndarray    # (384,)

    # Structural signals
    delegation_graph_adj: np.ndarray    # (100,) flat adjacency
    called_mask: np.ndarray             # (max_specialists,) binary

    # Scalar features
    step_count: int
    delegation_depth: int
    num_specialists_called: int
    max_specialists: int
    max_depth: int
    elapsed_ms: float
    sla_budget_ms: float
    phase: int  # 1, 2, or 3 (curriculum phase)

    def to_flat_vector(self) -> np.ndarray:
        """
        Flatten the full state to a 1D numpy array for the policy.
        This is the observation that the LSTM policy receives.
        """
        scalar_features = np.array([
            self.step_count / 10.0,
            self.delegation_depth / self.max_depth,
            self.num_specialists_called / self.max_specialists,
            self.elapsed_ms / max(self.sla_budget_ms, 1.0),
            float(self.phase) / 3.0,
            float(self.num_specialists_called > 0),
            float(self.delegation_depth == self.max_depth),
            float(self.elapsed_ms > self.sla_budget_ms * 0.8),
        ], dtype=np.float32)

        parts = [
            self.task_embedding.flatten(),
            self.roster_embeddings.flatten(),
            self.called_embeddings.flatten(),
            self.scratchpad_embedding.flatten(),
            self.delegation_graph_adj.flatten(),
            self.called_mask.flatten(),
            scalar_features,
        ]
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def observation_dim(max_specialists: int = 8) -> int:
        """Compute the flat observation dimension given max_specialists."""
        task       = 384
        roster     = max_specialists * 384
        called     = max_specialists * 384
        scratchpad = 384
        graph      = 100   # 10×10 adjacency
        mask       = max_specialists
        scalars    = 8
        return task + roster + called + scratchpad + graph + mask + scalars


def build_state(
    task_embedding: np.ndarray,
    registry,           # SpecialistRegistry
    called_ids: list[str],
    delegation_graph,   # DelegationGraph
    scratchpad,         # SharedScratchpad
    step_count: int,
    elapsed_ms: float,
    sla_budget_ms: float,
    max_specialists: int = 8,
    max_depth: int = 2,
    phase: int = 1,
    active_ids: list[str] | None = None,
) -> EpisodeState:
    """
    Factory function to build EpisodeState from all environment components.
    Called at each step by SpindleFlowEnv.

    active_ids: explicit per-episode roster (top-K by task similarity + any spawned
                specialists). When provided, replaces the default insertion-order slice.
    """
    all_ids = (list(active_ids) if active_ids is not None
               else registry.list_ids())[:max_specialists]

    # Roster embeddings matrix
    roster_matrix = np.zeros((max_specialists, 384), dtype=np.float32)
    for i, sid in enumerate(all_ids):
        if i >= max_specialists:
            break
        roster_matrix[i] = registry.get(sid).to_state_vector()

    # Called specialist embeddings
    called_matrix = np.zeros((max_specialists, 384), dtype=np.float32)
    called_mask = np.zeros(max_specialists, dtype=np.float32)
    for i, sid in enumerate(all_ids):
        if sid in called_ids and i < max_specialists:
            called_matrix[i] = registry.get(sid).to_state_vector()
            called_mask[i] = 1.0

    # Delegation graph adjacency vector
    adj_vector = np.array(
        delegation_graph.to_adjacency_vector(all_ids, max_size=10),
        dtype=np.float32,
    )

    # Scratchpad summary embedding
    scratchpad_emb = np.array(
        scratchpad.to_summary_vector(registry.embed_query),
        dtype=np.float32,
    )

    return EpisodeState(
        task_embedding=task_embedding,
        roster_embeddings=roster_matrix,
        called_embeddings=called_matrix,
        scratchpad_embedding=scratchpad_emb,
        delegation_graph_adj=adj_vector,
        called_mask=called_mask,
        step_count=step_count,
        delegation_depth=delegation_graph.depth,
        num_specialists_called=len(called_ids),
        max_specialists=max_specialists,
        max_depth=max_depth,
        elapsed_ms=elapsed_ms,
        sla_budget_ms=sla_budget_ms,
        phase=phase,
    )
