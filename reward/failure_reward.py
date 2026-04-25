"""Failure handling reward signals — partial credit for recovery."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class SpecialistStatus(Enum):
    SUCCESS      = "success"
    TIMEOUT      = "timeout"
    ERROR        = "error"
    FALLBACK_USED = "fallback_used"
    PARTIAL      = "partial"


@dataclass
class SpecialistResult:
    specialist_id: str
    status: SpecialistStatus
    output: str
    latency_ms: float
    fallback_used: bool = False


def compute_failure_penalty(results: list[SpecialistResult]) -> float:
    """Penalize for failed specialists. Reduce penalty if fallback worked."""
    penalty = 0.0
    for result in results:
        if result.status == SpecialistStatus.TIMEOUT:
            base = 0.3
            penalty += base * (0.3 if result.fallback_used else 1.0)
        elif result.status == SpecialistStatus.ERROR:
            base = 0.2
            penalty += base * (0.3 if result.fallback_used else 1.0)
    return min(penalty, 0.6)  # Cap total failure penalty


def compute_recovery_bonus(
    results: list[SpecialistResult],
    episode_completed: bool,
) -> float:
    """Bonus for successfully recovering from a failure."""
    failed_with_fallback = sum(
        1 for r in results
        if r.fallback_used and r.status != SpecialistStatus.ERROR
    )
    if failed_with_fallback > 0 and episode_completed:
        return 0.1 * failed_with_fallback
    return 0.0
