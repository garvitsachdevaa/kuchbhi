"""
ResolutionMemory — ε-greedy bandit over conflict resolution templates.

Tracks (conflict_type, template_key, quality_delta) outcomes and learns
which template produces the best quality improvements per conflict type.
No deep learning required — the arm count is small (4 types × N templates).
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ResolutionOutcome:
    conflict_type: str    # ConflictType.value string
    template_key:  str
    quality_delta: float  # specialist_score - baseline_score for the episode
    episode_idx:   int


class ResolutionBandit:
    """
    ε-greedy bandit that selects a resolution template for a given conflict type.
    Falls back to random selection until min_samples observations exist.

    Config keys (read from agents sub-dict of training config):
      resolution_bandit_epsilon       — exploration rate (default 0.15)
      resolution_bandit_min_samples   — minimum observations before exploiting (default 5)
    """

    def __init__(
        self,
        templates: dict[str, dict[str, str]],
        config: dict,
        memory_path: str,
    ):
        self._templates    = templates        # {ct_value_str: {template_key: template_str}}
        self._epsilon      = config.get("resolution_bandit_epsilon", 0.15)
        self._min_samples  = config.get("resolution_bandit_min_samples", 5)
        self._memory_path  = Path(memory_path)
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        # {conflict_type_str: {template_key: [quality_deltas]}}
        self._stats: dict[str, dict[str, list[float]]] = {}
        self._load()

    def _load(self) -> None:
        if not self._memory_path.exists():
            return
        for line in self._memory_path.read_text().splitlines():
            try:
                rec = ResolutionOutcome(**json.loads(line))
                (self._stats
                 .setdefault(rec.conflict_type, {})
                 .setdefault(rec.template_key, [])
                 .append(rec.quality_delta))
            except Exception:
                continue

    def select_template(self, conflict_type_str: str) -> str:
        """
        ε-greedy selection over available templates for this conflict type.
        Returns the template key (not the template text).
        Falls back to the first available key if the type is unknown.
        """
        available = list(self._templates.get(conflict_type_str, {}).keys())
        if not available:
            return "default"

        type_stats = self._stats.get(conflict_type_str, {})
        if random.random() < self._epsilon or not type_stats:
            return random.choice(available)

        scored = {
            k: sum(v) / len(v)
            for k, v in type_stats.items()
            if k in available and len(v) >= self._min_samples
        }
        if not scored:
            return random.choice(available)
        return max(scored, key=scored.__getitem__)

    def record_outcome(self, outcome: ResolutionOutcome) -> None:
        (self._stats
         .setdefault(outcome.conflict_type, {})
         .setdefault(outcome.template_key, [])
         .append(outcome.quality_delta))
        with open(self._memory_path, "a") as f:
            f.write(json.dumps(asdict(outcome)) + "\n")

    def arm_means(self) -> dict[str, dict[str, float]]:
        """Return current mean quality delta per (conflict_type, template_key)."""
        return {
            ct: {
                tk: sum(deltas) / len(deltas)
                for tk, deltas in tk_map.items()
                if deltas
            }
            for ct, tk_map in self._stats.items()
        }
