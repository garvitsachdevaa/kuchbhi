"""
Delegation trace — audit trail for regulated industries.
Every delegation decision is logged. generate_explanation() produces
human-readable audit text.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from env.delegation_graph import DelegationEdge


@dataclass
class DelegationTrace:
    """Complete audit record for one episode."""
    episode_id: str
    task_description: str
    task_complexity: str
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    delegation_edges: list[DelegationEdge] = field(default_factory=list)
    scratchpad_entries: list[dict] = field(default_factory=list)
    final_reward: float = 0.0
    approved_by_policy: bool = True

    def record_edge(self, edge: DelegationEdge) -> None:
        self.delegation_edges.append(edge)

    def record_scratchpad(self, author_id: str, content: str, step: int) -> None:
        self.scratchpad_entries.append({
            "author": author_id,
            "step": step,
            "content_preview": content[:200],
        })

    def generate_explanation(self) -> str:
        """
        Generate a human-readable audit trail.
        Suitable for compliance export.
        """
        lines = [
            "=== DELEGATION AUDIT TRAIL ===",
            f"Episode: {self.episode_id}",
            f"Time: {self.start_time}",
            f"Task: {self.task_description}",
            f"Complexity: {self.task_complexity}",
            f"Final Reward: {self.final_reward:.3f}",
            "",
            "Delegation Sequence:",
        ]

        for i, edge in enumerate(self.delegation_edges):
            lines.append(
                f"  Step {i+1}: {edge.caller_id} -> {edge.callee_id} "
                f"[mode: {edge.delegation_mode}]"
            )

        lines.extend([
            "",
            f"Total specialists called: {len(self.delegation_edges)}",
            f"Max delegation depth reached: "
            f"{max((e.depth for e in self.delegation_edges), default=0)}",
            "=== END AUDIT TRAIL ===",
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task": self.task_description,
            "complexity": self.task_complexity,
            "start_time": self.start_time,
            "delegation_steps": [
                {
                    "caller": e.caller_id,
                    "callee": e.callee_id,
                    "mode": e.delegation_mode,
                    "depth": e.depth,
                }
                for e in self.delegation_edges
            ],
            "reward": self.final_reward,
        }
