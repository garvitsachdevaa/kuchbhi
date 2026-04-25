"""
SB3 callback that periodically improves specialist prompts using
SpecialistFinetuner + SpecialistMemory.

Wired into model.learn() alongside CheckpointCallback in train.py.
Triggers every `improve_every_n_episodes` completed episodes.
"""

from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback


class SpecialistImprovementCallback(BaseCallback):
    """
    After every `improve_every_n_episodes` episodes, run the finetuner over
    all specialists that have enough memory entries and below-threshold reward.
    Also saves the memory file after each improvement pass.
    """

    def __init__(self, improve_every_n_episodes: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._improve_every = improve_every_n_episodes
        self._episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        self._episode_count += int(sum(dones))
        if self._episode_count >= self._improve_every:
            self._episode_count = 0
            self._run_improvement()
        return True

    def _run_improvement(self) -> None:
        from agents.specialist_finetuner import SpecialistFinetuner

        env = self._get_env()
        if env is None:
            return

        memory = getattr(env, "specialist_memory", None)
        registry = getattr(env, "registry", None)
        if memory is None or registry is None:
            return

        cfg = getattr(env, "config", {})
        si_cfg = cfg.get("specialist_improvement", {})
        min_entries = si_cfg.get("min_entries_to_improve", 10)
        threshold = si_cfg.get("improve_avg_reward_threshold", 0.70)

        finetuner = SpecialistFinetuner(
            min_entries=min_entries,
            improve_threshold=threshold,
        )
        n = finetuner.improve_all(registry, memory)
        memory.save()
        if self.verbose and n > 0:
            print(f"[SpecialistImprovementCallback] Improved {n} specialist(s).")

    def _get_env(self):
        """Unwrap VecNormalize → DummyVecEnv → first env."""
        try:
            venv = self.training_env
            # VecNormalize wraps venv; DummyVecEnv has .envs
            inner = getattr(venv, "venv", venv)
            envs = getattr(inner, "envs", None)
            if envs:
                return envs[0]
        except Exception:
            pass
        return None
