"""
Episode Tier Lock — ensures both baseline and specialist output are scored
through the SAME tier. Prevents the circular dependency bug from v3.
"""

from __future__ import annotations
import random
from enum import IntEnum
from dataclasses import dataclass


class RewardTier(IntEnum):
    TIER_0 = 0  # Free structural checks
    TIER_1 = 1  # Embedding similarity
    TIER_2 = 2  # Small LLM micro-judge (GPT-4o-mini)
    TIER_3 = 3  # Full LLM-as-judge (checkpoints only)


def _load_tier_config() -> tuple[dict, dict]:
    """Load tier_map and tier2_sample_rates from training_config.yaml at import time."""
    import yaml, os
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "training_config.yaml"
    )
    try:
        with open(config_path) as f:
            reward_cfg = yaml.safe_load(f).get("reward", {})
        tier_map_raw = reward_cfg.get("tier_map", {
            "atomic": 0, "simple": 1, "moderate": 1, "complex": 2, "enterprise": 2,
        })
        tier_map = {k: RewardTier(v) for k, v in tier_map_raw.items()}
        sample_rates = reward_cfg.get("tier2_sample_rates", {
            "moderate": 0.30, "complex": 1.00, "enterprise": 1.00,
        })
        return tier_map, sample_rates
    except Exception:
        return (
            {"atomic": RewardTier.TIER_0, "simple": RewardTier.TIER_1,
             "moderate": RewardTier.TIER_1, "complex": RewardTier.TIER_2,
             "enterprise": RewardTier.TIER_2},
            {"moderate": 0.30, "complex": 1.00, "enterprise": 1.00},
        )


TIER_MAP, TIER2_SAMPLE_RATES = _load_tier_config()


@dataclass
class EpisodeTierLock:
    """
    Locked once at episode start. Both generalist and specialist outputs
    are scored through this exact tier. No drift.
    """
    complexity_class: str
    locked_tier: RewardTier
    tier2_sample_rate: float

    @classmethod
    def for_task(cls, complexity_class: str) -> "EpisodeTierLock":
        tier = TIER_MAP.get(complexity_class, RewardTier.TIER_1)
        sample_rate = TIER2_SAMPLE_RATES.get(complexity_class, 0.0)
        if complexity_class == "moderate" and random.random() < sample_rate:
            tier = RewardTier.TIER_2
        return cls(
            complexity_class=complexity_class,
            locked_tier=tier,
            tier2_sample_rate=sample_rate,
        )
