"""
Path consistency tracking with Dirichlet prior.
Non-zero from Episode 1, avoids cold-start problem from v3.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict


class PathConsistencyTracker:
    """
    Tracks how consistently the policy routes the same task type.
    Uses a Dirichlet prior (alpha=1.0) so the bonus is non-zero from episode 1.
    """

    DIRICHLET_ALPHA = 1.0

    def __init__(self, specialist_ids: list[str]):
        self.specialist_ids = specialist_ids
        self._task_path_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def record_path(self, task_class: str, delegation_path: list) -> None:
        """Record the delegation path used for a task class."""
        path_key = self._path_to_key(delegation_path)
        self._task_path_counts[task_class][path_key] += 1

    def consistency_score(
        self, delegation_path: list, task_class: str
    ) -> float:
        """
        Score how consistent this path is with previous paths for this task class.
        Returns 0.0–1.0. Non-zero from episode 1 due to Dirichlet prior.
        """
        path_key = self._path_to_key(delegation_path)
        counts = self._task_path_counts.get(task_class, {})

        # Add Dirichlet prior counts
        all_paths = set(counts.keys()) | {path_key}
        pseudo_counts = {p: counts.get(p, 0) + self.DIRICHLET_ALPHA for p in all_paths}
        total = sum(pseudo_counts.values())

        return float(pseudo_counts[path_key] / total)

    def _path_to_key(self, delegation_path: list) -> str:
        """Convert a delegation path to a hashable string key."""
        if not delegation_path:
            return "empty"
        parts = []
        for edge in delegation_path:
            if hasattr(edge, "callee_id"):
                parts.append(edge.callee_id)
            elif isinstance(edge, dict):
                parts.append(edge.get("callee_id", "?"))
        return "->".join(parts)
