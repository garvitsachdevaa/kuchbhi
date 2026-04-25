"""
Conflict Resolver — handles contradictions between specialist outputs.
Templates are loaded from configs/conflict_templates.yaml.
Template selection is bandit-guided: each conflict type has multiple named
strategies; ResolutionBandit picks the one with the highest historical
quality delta (ε-greedy, falls back to random when data is sparse).
"""

from __future__ import annotations
import yaml
from reward.conflict_reward import Conflict, ConflictType
from agents.resolution_memory import ResolutionBandit, ResolutionOutcome


def _load_templates(
    templates_path: str = "configs/conflict_templates.yaml",
) -> dict[ConflictType, dict[str, str]]:
    try:
        with open(templates_path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"conflict_templates.yaml not found at {templates_path}. "
            "This file is required — do not delete it."
        )
    mapping = {
        "TECHNICAL": ConflictType.TECHNICAL,
        "FACTUAL":   ConflictType.FACTUAL,
        "PRIORITY":  ConflictType.PRIORITY,
        "SCOPE":     ConflictType.SCOPE,
    }
    return {mapping[k]: v for k, v in raw.items() if k in mapping}


def _templates_by_str(
    templates: dict[ConflictType, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Convert ConflictType-keyed dict to value-string-keyed for the bandit."""
    return {ct.value: v for ct, v in templates.items()}


class ConflictResolver:
    """
    Mediates conflicts between specialist outputs.
    Selects resolution templates via a ε-greedy bandit; learns which strategy
    produces the best quality deltas over training.
    """

    def __init__(
        self,
        templates_path: str = "configs/conflict_templates.yaml",
        config: dict | None = None,
        memory_path: str = "data/resolution_memory.jsonl",
    ):
        self._templates = _load_templates(templates_path)
        agents_cfg = (config or {}).get("agents", {})
        self._bandit = ResolutionBandit(
            templates=_templates_by_str(self._templates),
            config=agents_cfg,
            memory_path=memory_path,
        )
        # Tracks (conflict_type_str, template_key) pairs used this episode
        self._episode_selections: list[tuple[str, str]] = []

    def resolve(self, conflict: Conflict, results: list) -> str:
        """Select and apply a resolution template via the bandit."""
        ct_str = conflict.conflict_type.value
        template_key = self._bandit.select_template(ct_str)

        type_templates = self._templates.get(conflict.conflict_type, {})
        template = type_templates.get(template_key) or next(
            iter(type_templates.values()),
            "Conflict detected between {a} and {b}. Prefer the more specific answer.",
        )
        resolution = template.format(
            a=conflict.agent_a,
            b=conflict.agent_b,
            a_use_case="performance-critical paths",
            b_use_case="general usage",
        )
        conflict.resolved = True
        self._episode_selections.append((ct_str, template_key))
        return resolution

    def resolve_all(self, conflicts: list[Conflict], results: list) -> list[str]:
        """Resolve all conflicts. Returns list of resolution strings."""
        return [self.resolve(c, results) for c in conflicts]

    def record_episode_outcome(
        self, quality_delta: float, episode_idx: int
    ) -> None:
        """
        Call at episode end to record how well the resolutions performed.
        Clears episode selections after recording.
        """
        for ct, tk in self._episode_selections:
            self._bandit.record_outcome(ResolutionOutcome(
                conflict_type=ct,
                template_key=tk,
                quality_delta=quality_delta,
                episode_idx=episode_idx,
            ))
        self._episode_selections = []
