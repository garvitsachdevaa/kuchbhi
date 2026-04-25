"""Tests for delegation graph DAG enforcement."""
import pytest
from env.delegation_graph import DelegationGraph


def test_no_self_delegation():
    g = DelegationGraph(max_depth=2)
    g.add_root("orchestrator")
    assert not g.can_delegate("orchestrator", "orchestrator")


def test_basic_delegation():
    g = DelegationGraph(max_depth=2)
    g.add_root("orchestrator")
    assert g.can_delegate("orchestrator", "frontend_react")
    g.record_delegation("orchestrator", "frontend_react", "sequential")
    assert "frontend_react" in g.get_called_specialists()


def test_cycle_prevention():
    g = DelegationGraph(max_depth=3)
    g.add_root("orchestrator")
    g.record_delegation("orchestrator", "a", "sequential")
    g.record_delegation("a", "b", "sequential")
    # b -> orchestrator should be blocked (cycle)
    assert not g.can_delegate("b", "orchestrator")
    # b -> a should be blocked (cycle)
    assert not g.can_delegate("b", "a")


def test_depth_enforcement():
    g = DelegationGraph(max_depth=2)
    g.add_root("orchestrator")
    g.record_delegation("orchestrator", "a", "sequential")
    g.record_delegation("a", "b", "sequential")
    # depth 3 would exceed max_depth=2
    assert not g.can_delegate("b", "c")


def test_adjacency_vector():
    g = DelegationGraph(max_depth=2)
    g.add_root("orchestrator")
    g.record_delegation("orchestrator", "frontend_react", "parallel")
    all_ids = ["orchestrator", "frontend_react", "backend_api"]
    vec = g.to_adjacency_vector(all_ids, max_size=3)
    assert len(vec) == 9  # 3x3
    assert vec[1] == 1.0  # orchestrator->frontend_react edge
