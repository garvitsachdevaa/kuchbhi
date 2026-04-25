"""Tests for action decoder and policy components."""
import pytest
import numpy as np
from env.action_space import ActionDecoder, MetaAction, DelegationMode


def test_action_decoder_stop():
    decoder = ActionDecoder(["a", "b", "c"], max_specialists=3)
    action = np.zeros(decoder.get_action_dim(), dtype=np.float32)
    action[0] = 1.0  # STOP
    factored = decoder.decode(action)
    assert factored.meta_action == MetaAction.STOP
    assert factored.is_terminal()


def test_action_decoder_call_specialist():
    ids = ["frontend_react", "backend_api", "database_architect"]
    decoder = ActionDecoder(ids, max_specialists=3)
    action = np.zeros(decoder.get_action_dim(), dtype=np.float32)
    action[0] = 0.0   # CALL_SPECIALIST
    action[1] = 1.0   # Select frontend_react
    factored = decoder.decode(action)
    assert factored.meta_action == MetaAction.CALL_SPECIALIST
    assert "frontend_react" in factored.specialist_ids


def test_specialist_mask():
    ids = ["a", "b", "c"]
    decoder = ActionDecoder(ids, max_specialists=3)
    mask = decoder.build_specialist_mask(["b"])
    assert mask[0] == 0.0
    assert mask[1] == 1.0
    assert mask[2] == 0.0


def test_action_dim():
    decoder = ActionDecoder(["a", "b"], max_specialists=2)
    assert decoder.get_action_dim() == 2 + 6
