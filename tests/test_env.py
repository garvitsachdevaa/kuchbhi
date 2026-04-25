"""Smoke tests for the main environment."""
import pytest
import numpy as np
from env.spindleflow_env import SpindleFlowEnv


@pytest.fixture
def env():
    e = SpindleFlowEnv(
        config_path="configs/training_config.yaml",
        catalog_path="configs/specialist_catalog.yaml",
        use_real_spindleflow=False,
        phase=1,
    )
    yield e
    e.close()


def test_env_reset(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape


def test_env_step_stop(env):
    obs, _ = env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0  # STOP action
    obs2, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)


def test_env_step_call_specialist(env):
    obs, _ = env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 0.0  # CALL_SPECIALIST
    action[1] = 1.0  # Select first specialist
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == env.observation_space.shape


def test_observation_space_shape(env):
    from env.state import EpisodeState
    expected_dim = EpisodeState.observation_dim(env.max_specialists)
    assert env.observation_space.shape == (expected_dim,)


def test_episode_runs_to_completion(env):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 15:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert done  # Episode must terminate within max_steps
