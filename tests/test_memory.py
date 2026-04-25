"""Tests for SpecialistMemory, ResolutionBandit, and SpawnMemory."""
import numpy as np
import pytest

from agents.specialist_memory import SpecialistMemory
from agents.resolution_memory import ResolutionBandit, ResolutionOutcome
from training.spawn_memory import SpawnMemory, SpawnRecord


# ── SpecialistMemory ──────────────────────────────────────────────────────────

def test_specialist_memory_record_and_retrieve(tmp_path):
    mem = SpecialistMemory(path=str(tmp_path / "mem.json"))
    mem.record("spec_a", "build an API", "Here is the API design.", reward=0.8)
    mem.record("spec_a", "write tests", "Here are the tests.", reward=0.5)
    assert mem.count("spec_a") == 2
    top = mem.get_top_examples("spec_a", n=2)
    assert top[0].reward == 0.8
    assert top[1].reward == 0.5


def test_specialist_memory_eviction(tmp_path):
    mem = SpecialistMemory(path=str(tmp_path / "mem.json"))
    mem.MAX_PER_SPECIALIST = 5
    for i in range(7):
        mem.record("spec_b", f"task {i}", f"output {i}", reward=float(i))
    # Lowest-reward entries should be evicted; only 5 remain
    assert mem.count("spec_b") == 5
    # Remaining entries should all be the 5 highest-reward ones (rewards 2–6)
    rewards = {e.reward for e in mem.get_top_examples("spec_b", n=5)}
    assert rewards == {2.0, 3.0, 4.0, 5.0, 6.0}


def test_specialist_memory_top_examples_sorted(tmp_path):
    mem = SpecialistMemory(path=str(tmp_path / "mem.json"))
    for reward in [0.3, 0.9, 0.1, 0.7]:
        mem.record("spec_c", "task", "output", reward=reward)
    top = mem.get_top_examples("spec_c", n=4)
    assert top[0].reward == 0.9
    assert top[-1].reward == 0.1


def test_specialist_memory_avg_reward(tmp_path):
    mem = SpecialistMemory(path=str(tmp_path / "mem.json"))
    mem.record("spec_d", "t", "o", reward=0.4)
    mem.record("spec_d", "t", "o", reward=0.6)
    assert abs(mem.avg_reward("spec_d") - 0.5) < 1e-6


def test_specialist_memory_empty_specialist(tmp_path):
    mem = SpecialistMemory(path=str(tmp_path / "mem.json"))
    assert mem.count("nobody") == 0
    assert mem.avg_reward("nobody") == 0.0
    assert mem.get_top_examples("nobody") == []


# ── ResolutionBandit ──────────────────────────────────────────────────────────

_TEMPLATES = {
    "technical": {"standard": "Use {a}.", "defer_to_a": "Defer to {a}."},
    "factual":   {"recency": "Use recent claim from {a}."},
}


def test_resolution_bandit_returns_valid_key(tmp_path):
    bandit = ResolutionBandit(
        templates=_TEMPLATES,
        config={"resolution_bandit_epsilon": 0.0, "resolution_bandit_min_samples": 1},
        memory_path=str(tmp_path / "res.jsonl"),
    )
    key = bandit.select_template("technical")
    assert key in _TEMPLATES["technical"]


def test_resolution_bandit_exploits_best_arm(tmp_path):
    bandit = ResolutionBandit(
        templates=_TEMPLATES,
        config={"resolution_bandit_epsilon": 0.0, "resolution_bandit_min_samples": 2},
        memory_path=str(tmp_path / "res.jsonl"),
    )
    # Seed defer_to_a with high deltas, standard with low
    for _ in range(3):
        bandit.record_outcome(ResolutionOutcome("technical", "defer_to_a", 0.9, 0))
        bandit.record_outcome(ResolutionOutcome("technical", "standard",   0.1, 0))
    assert bandit.select_template("technical") == "defer_to_a"


def test_resolution_bandit_random_when_insufficient_samples(tmp_path):
    bandit = ResolutionBandit(
        templates=_TEMPLATES,
        config={"resolution_bandit_epsilon": 0.0, "resolution_bandit_min_samples": 10},
        memory_path=str(tmp_path / "res.jsonl"),
    )
    # Only 2 samples — below min_samples of 10, so should still return a valid key
    bandit.record_outcome(ResolutionOutcome("technical", "standard", 0.8, 0))
    bandit.record_outcome(ResolutionOutcome("technical", "standard", 0.7, 0))
    key = bandit.select_template("technical")
    assert key in _TEMPLATES["technical"]


def test_resolution_bandit_arm_means(tmp_path):
    bandit = ResolutionBandit(
        templates=_TEMPLATES,
        config={},
        memory_path=str(tmp_path / "res.jsonl"),
    )
    bandit.record_outcome(ResolutionOutcome("technical", "standard", 0.4, 0))
    bandit.record_outcome(ResolutionOutcome("technical", "standard", 0.6, 0))
    means = bandit.arm_means()
    assert abs(means["technical"]["standard"] - 0.5) < 1e-6


def test_resolution_bandit_unknown_type_returns_default(tmp_path):
    bandit = ResolutionBandit(
        templates=_TEMPLATES,
        config={},
        memory_path=str(tmp_path / "res.jsonl"),
    )
    assert bandit.select_template("nonexistent_type") == "default"


# ── SpawnMemory ───────────────────────────────────────────────────────────────

def _make_record(task_emb, reward=0.5, sid="spec_x"):
    return SpawnRecord(
        task_embedding=task_emb.tolist(),
        task_description="test task",
        specialist_id=sid,
        specialist_role="Test Role",
        specialist_desc="A test specialist.",
        episode_reward=reward,
        pre_spawn_sim=0.3,
        post_spawn_sim=0.7,
        episode_idx=0,
    )


def test_spawn_memory_record_and_size(tmp_path):
    mem = SpawnMemory(path=str(tmp_path / "spawn.jsonl"))
    emb = np.random.rand(384).astype(np.float32)
    mem.record(_make_record(emb))
    assert mem.size == 1


def test_spawn_memory_retrieve_similar_ordering(tmp_path):
    mem = SpawnMemory(path=str(tmp_path / "spawn.jsonl"))
    base = np.ones(384, dtype=np.float32)
    # Record two spawns: one very similar to base, one orthogonal
    similar_emb = base + np.random.rand(384).astype(np.float32) * 0.01
    orthogonal_emb = np.zeros(384, dtype=np.float32)
    orthogonal_emb[0] = 1.0
    mem.record(_make_record(similar_emb, reward=0.5, sid="similar"))
    mem.record(_make_record(orthogonal_emb, reward=0.5, sid="orthogonal"))
    results = mem.retrieve_similar(base / np.linalg.norm(base), top_k=2)
    assert results[0].specialist_id == "similar"


def test_spawn_memory_min_reward_filter(tmp_path):
    mem = SpawnMemory(path=str(tmp_path / "spawn.jsonl"))
    emb = np.ones(384, dtype=np.float32)
    mem.record(_make_record(emb, reward=0.1, sid="low"))
    mem.record(_make_record(emb, reward=0.8, sid="high"))
    results = mem.retrieve_similar(emb / np.linalg.norm(emb), top_k=5, min_reward=0.5)
    ids = [r.specialist_id for r in results]
    assert "high" in ids
    assert "low" not in ids


def test_spawn_memory_eviction_keeps_highest_reward(tmp_path):
    mem = SpawnMemory(path=str(tmp_path / "spawn.jsonl"), max_entries=3)
    emb = np.ones(384, dtype=np.float32)
    for reward in [0.1, 0.9, 0.5, 0.8]:
        mem.record(_make_record(emb, reward=reward))
    assert mem.size == 3
    rewards = {r.episode_reward for r in mem._records}
    assert rewards == {0.9, 0.8, 0.5}
