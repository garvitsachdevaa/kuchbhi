"""
Delegation Graph — Directed Acyclic Graph enforcement for delegation chains.

Prevents: A → B → A (infinite loops)
Prevents: A → B → C → A (indirect cycles)
Enforces: Maximum delegation depth budget
Provides: Action masking for valid next-call candidates
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional


@dataclass
class DelegationEdge:
    caller_id: str
    callee_id: str
    depth: int
    delegation_mode: str
    step: int


class DelegationGraph:
    """
    Enforces delegation as a DAG. No cycles, no depth violations.

    Design: Built incrementally during an episode. At each step,
    before executing an action, the policy checks `can_delegate(caller, callee)`.
    If False, the action is masked to zero probability.
    """

    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        self._edges: list[DelegationEdge] = []
        self._adj: dict[str, set[str]] = defaultdict(set)  # caller → callees
        self._depth_map: dict[str, int] = {}  # node_id → depth from root
        self._current_depth: int = 0
        self._step: int = 0

    def reset(self) -> None:
        """Reset graph for a new episode."""
        self._edges.clear()
        self._adj.clear()
        self._depth_map.clear()
        self._current_depth = 0
        self._step = 0

    def add_root(self, orchestrator_id: str) -> None:
        """Register the orchestrator as the root node at depth 0."""
        self._depth_map[orchestrator_id] = 0

    def can_delegate(self, caller_id: str, callee_id: str) -> bool:
        """
        Check if caller CAN delegate to callee.
        Returns False if:
        - Adding this edge would create a cycle
        - callee is already at max_depth
        - caller == callee (self-delegation)
        """
        if caller_id == callee_id:
            return False

        caller_depth = self._depth_map.get(caller_id, 0)
        proposed_callee_depth = caller_depth + 1

        if proposed_callee_depth > self.max_depth:
            return False

        if self._would_create_cycle(caller_id, callee_id):
            return False

        return True

    def _would_create_cycle(self, caller_id: str, callee_id: str) -> bool:
        """
        Check if adding edge (caller → callee) would create a cycle.
        Uses DFS from callee to see if we can reach caller.
        """
        if callee_id not in self._adj:
            return False  # callee has no outgoing edges yet

        visited = set()
        stack = deque([callee_id])
        while stack:
            node = stack.pop()
            if node == caller_id:
                return True
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self._adj.get(node, set()):
                stack.append(neighbor)
        return False

    def record_delegation(
        self,
        caller_id: str,
        callee_id: str,
        delegation_mode: str,
    ) -> None:
        """
        Record a delegation edge after validation.
        Call ONLY after `can_delegate()` returned True.
        """
        if not self.can_delegate(caller_id, callee_id):
            raise ValueError(
                f"Invalid delegation: {caller_id} → {callee_id} "
                f"(would create cycle or exceed depth)"
            )

        caller_depth = self._depth_map.get(caller_id, 0)
        callee_depth = caller_depth + 1

        self._adj[caller_id].add(callee_id)
        self._depth_map[callee_id] = callee_depth
        self._current_depth = max(self._current_depth, callee_depth)

        edge = DelegationEdge(
            caller_id=caller_id,
            callee_id=callee_id,
            depth=callee_depth,
            delegation_mode=delegation_mode,
            step=self._step,
        )
        self._edges.append(edge)
        self._step += 1

    def get_valid_callees(
        self, caller_id: str, all_specialist_ids: list[str]
    ) -> list[str]:
        """
        Return the list of specialist IDs that caller can still delegate to.
        Used for action masking in the policy.
        """
        return [
            sid for sid in all_specialist_ids
            if self.can_delegate(caller_id, sid)
        ]

    def get_called_specialists(self) -> list[str]:
        """Return all specialists called so far this episode."""
        called = set()
        for edge in self._edges:
            called.add(edge.callee_id)
        return list(called)

    def get_delegation_path(self) -> list[DelegationEdge]:
        """Return the full delegation path for this episode."""
        return list(self._edges)

    @property
    def depth(self) -> int:
        return self._current_depth

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def to_adjacency_vector(
        self, all_ids: list[str], max_size: int = 10
    ) -> list[float]:
        """
        Encode the delegation graph as a flat adjacency vector for the policy.
        Shape: (max_size * max_size,) — padded with zeros.

        This replaces the GNN layer from the original v3 design.
        An MLP operating on this vector is sufficient for the hackathon demo.
        Production would use a proper GNN.
        """
        n = min(len(all_ids), max_size)
        id_to_idx = {sid: i for i, sid in enumerate(all_ids[:n])}
        matrix = [[0.0] * n for _ in range(n)]

        for edge in self._edges:
            if edge.caller_id in id_to_idx and edge.callee_id in id_to_idx:
                i = id_to_idx[edge.caller_id]
                j = id_to_idx[edge.callee_id]
                matrix[i][j] = 1.0

        flat = []
        for row in matrix:
            flat.extend(row)

        target_len = max_size * max_size
        flat.extend([0.0] * (target_len - len(flat)))
        return flat[:target_len]

    def is_auditable(self) -> bool:
        """
        Returns True if the delegation path has a clear, explainable structure.
        Criteria: all edges recorded, no cycles detected, depth ≤ max_depth.
        """
        return (
            len(self._edges) > 0
            and self._current_depth <= self.max_depth
        )
