"""Latency SLA reward component."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LatencySLAConfig:
    budget_ms: float = 10000.0  # 10 second default SLA
    weight: float = 0.05


def compute_latency_penalty(
    elapsed_ms: float,
    config: LatencySLAConfig,
) -> float:
    """
    Compute latency penalty as a fraction of the SLA budget.
    Returns 0 if within SLA, increasing penalty if over.
    """
    if elapsed_ms <= config.budget_ms:
        return 0.0
    overage_fraction = (elapsed_ms - config.budget_ms) / config.budget_ms
    return config.weight * min(overage_fraction, 1.0)
