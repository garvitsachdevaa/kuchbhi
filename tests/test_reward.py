"""Tests for reward system."""
import pytest
from reward.tier_lock import EpisodeTierLock, RewardTier
from reward.failure_reward import (
    SpecialistResult, SpecialistStatus,
    compute_failure_penalty, compute_recovery_bonus,
)
from reward.consistency_tracker import PathConsistencyTracker


def test_tier_lock_same_for_atomic():
    lock = EpisodeTierLock.for_task("atomic")
    assert lock.locked_tier == RewardTier.TIER_0


def test_tier_lock_same_for_complex():
    lock = EpisodeTierLock.for_task("complex")
    assert lock.locked_tier == RewardTier.TIER_2


def test_failure_penalty_with_fallback():
    results = [
        SpecialistResult("a", SpecialistStatus.TIMEOUT, "", 8000, fallback_used=True),
    ]
    penalty = compute_failure_penalty(results)
    assert penalty < 0.3  # Reduced because fallback was used


def test_failure_penalty_no_fallback():
    results = [
        SpecialistResult("a", SpecialistStatus.TIMEOUT, "", 8000, fallback_used=False),
    ]
    penalty = compute_failure_penalty(results)
    assert penalty == pytest.approx(0.3)


def test_consistency_nonzero_from_start():
    """Dirichlet prior ensures non-zero consistency from episode 1."""
    tracker = PathConsistencyTracker(specialist_ids=["a", "b", "c"])
    # No recorded paths yet — score should still be > 0
    score = tracker.consistency_score([], "simple")
    assert score > 0.0


def test_recovery_bonus():
    results = [
        SpecialistResult("a", SpecialistStatus.TIMEOUT, "fallback output", 3000, fallback_used=True),
    ]
    bonus = compute_recovery_bonus(results, episode_completed=True)
    assert bonus > 0.0


def test_conflict_detection_no_registry():
    """detect_conflicts works without a registry (keyword fallback only)."""
    from reward.conflict_reward import detect_conflicts
    results = [
        SpecialistResult("a", SpecialistStatus.SUCCESS, "Use PostgreSQL for storage", 1000),
        SpecialistResult("b", SpecialistStatus.SUCCESS, "Use MongoDB for storage", 1000),
    ]
    # No registry passed — should still work, returns empty list (no pairs provided)
    conflicts = detect_conflicts(results)
    assert isinstance(conflicts, list)


def test_conflict_detection_with_keyword_pairs():
    """detect_conflicts uses provided contradiction pairs correctly."""
    from reward.conflict_reward import detect_conflicts
    results = [
        SpecialistResult("a", SpecialistStatus.SUCCESS, "Use PostgreSQL for storage", 1000),
        SpecialistResult("b", SpecialistStatus.SUCCESS, "Use MongoDB for storage", 1000),
    ]
    conflicts = detect_conflicts(
        results,
        contradiction_pairs=[("postgresql", "mongodb")]
    )
    assert len(conflicts) == 1
    assert conflicts[0].agent_a == "a"
    assert conflicts[0].agent_b == "b"
