"""
OpenEnv wrapper — registers SpindleFlowEnv as an OpenEnv-compatible environment.

HACKATHON REQUIREMENT: OpenEnv (latest release) must be used.
This module makes SpindleFlowEnv discoverable and instantiable via the
OpenEnv registry, satisfying the minimum submission requirement.

Usage:
    import env.openenv_wrapper  # triggers registration
    import openenv
    env = openenv.make("SpindleFlow-v0")
"""

from __future__ import annotations

try:
    import openenv
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False
    print(
        "[OpenEnvWrapper] WARNING: openenv package not found. "
        "Run: pip install openenv\n"
        "This is a REQUIRED hackathon dependency."
    )

from env.spindleflow_env import SpindleFlowEnv


def make_spindleflow_env(**kwargs):
    """Factory function for OpenEnv registry."""
    return SpindleFlowEnv(**kwargs)


if _OPENENV_AVAILABLE:
    # Register with OpenEnv so `openenv.make("SpindleFlow-v0")` works
    try:
        openenv.register(
            id="SpindleFlow-v0",
            entry_point=make_spindleflow_env,
            kwargs={
                "config_path": "configs/training_config.yaml",
                "catalog_path": "configs/specialist_catalog.yaml",
                "use_real_spindleflow": False,
                "phase": 1,
            },
        )
        print("[OpenEnvWrapper] >> SpindleFlow-v0 registered with OpenEnv")
    except Exception as e:
        # openenv API may differ across versions — fall back gracefully
        print(f"[OpenEnvWrapper] Registration warning: {e}")
        print("[OpenEnvWrapper] Verify openenv version: pip show openenv")


def verify_openenv_compliance() -> bool:
    """
    Verify that the environment meets OpenEnv compliance.
    Called during Step 1 checklist verification.
    """
    if not _OPENENV_AVAILABLE:
        print("[FAIL] openenv not installed -- REQUIRED for hackathon submission")
        return False

    try:
        env = SpindleFlowEnv(
            config_path="configs/training_config.yaml",
            catalog_path="configs/specialist_catalog.yaml",
            use_real_spindleflow=False,
            phase=1,
        )
        obs, info = env.reset()
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        env.close()
        print("[PASS] OpenEnv compliance check passed (reset/step/close cycle OK)")
        return True
    except Exception as e:
        print(f"[FAIL] OpenEnv compliance check failed: {e}")
        return False
