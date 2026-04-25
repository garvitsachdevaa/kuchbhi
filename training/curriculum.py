"""
CurriculumManager — performance-gated phase advancement.

Phases advance when rolling_mean_reward >= phase_advance_threshold,
not after a fixed episode count. Thresholds and window size come from config.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import yaml


@dataclass
class CurriculumPhase:
    phase: int
    name: str
    episode_budget: int
    task_types: list[str]
    enable_tier2: bool
    enable_tier3: bool


class CurriculumManager:
    """
    Tracks curriculum progress and transitions between phases.
    Advances when the rolling mean reward over the last N episodes
    exceeds a configurable threshold — not after a fixed episode count.
    """

    _PHASE_NAMES = {
        1: "Simple Delegation",
        2: "Moderate Tasks + Conflict",
        3: "Complex + Enterprise",
    }
    _TIER2_PHASES = {2, 3}
    _TIER3_PHASES = {3}

    def __init__(self, config_path: str = "configs/training_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["curriculum"]

        # Performance-gated advancement parameters
        self._window_size   = cfg.get("phase_advance_window", 50)
        self._thresholds    = {
            1: cfg.get("phase1_advance_threshold", 0.30),
            2: cfg.get("phase2_advance_threshold", 0.50),
        }
        self._min_episodes  = cfg.get("phase_min_episodes", 100)

        # Task types still read from config (used by TaskBank)
        self._phase_task_types = {
            1: cfg.get("phase1_task_types", ["atomic", "simple"]),
            2: cfg.get("phase2_task_types", ["moderate"]),
            3: cfg.get("phase3_task_types", ["complex", "enterprise"]),
        }
        # Legacy budget fields — kept for get_current_phase() / progress_str()
        self._phase_budgets = {
            1: cfg.get("phase1_episodes", 200),
            2: cfg.get("phase2_episodes", 400),
            3: cfg.get("phase3_episodes", 600),
        }

        self.current_phase      = 1
        self.episodes_in_phase  = 0
        self.total_episodes     = 0
        self._reward_window: deque[float] = deque(maxlen=self._window_size)

    def on_episode_end(self, episode_reward: float = 0.0) -> bool:
        """
        Called after each episode with the terminal reward.
        Returns True if the phase advanced.
        """
        self.total_episodes    += 1
        self.episodes_in_phase += 1
        self._reward_window.append(episode_reward)

        if (
            self.current_phase < 3
            and self.episodes_in_phase >= self._min_episodes
            and len(self._reward_window) >= self._window_size
        ):
            rolling_mean = sum(self._reward_window) / len(self._reward_window)
            threshold    = self._thresholds.get(self.current_phase, float("inf"))
            if rolling_mean >= threshold:
                self.current_phase     += 1
                self.episodes_in_phase  = 0
                self._reward_window.clear()
                print(
                    f"\n[Curriculum] >> Advanced to Phase {self.current_phase} "
                    f"(rolling mean {rolling_mean:.3f} >= {threshold:.3f})"
                )
                return True
        return False

    @property
    def phase(self) -> int:
        return self.current_phase

    def rolling_mean(self) -> float:
        if not self._reward_window:
            return 0.0
        return sum(self._reward_window) / len(self._reward_window)

    def get_current_phase(self) -> CurriculumPhase:
        p = self.current_phase
        return CurriculumPhase(
            phase=p,
            name=self._PHASE_NAMES[p],
            episode_budget=self._phase_budgets[p],
            task_types=self._phase_task_types[p],
            enable_tier2=p in self._TIER2_PHASES,
            enable_tier3=p in self._TIER3_PHASES,
        )

    def progress_str(self) -> str:
        threshold = self._thresholds.get(self.current_phase, "—")
        return (
            f"Phase {self.current_phase}/3 | "
            f"Rolling mean: {self.rolling_mean():.3f} / {threshold} | "
            f"Episodes in phase: {self.episodes_in_phase}"
        )
