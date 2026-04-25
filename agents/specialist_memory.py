"""
Specialist Memory — records (task, output, reward) tuples per specialist.
Persisted to JSON so memory survives training restarts.
Used by SpecialistFinetuner to evolve specialist system prompts.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MemoryEntry:
    specialist_id: str
    task: str
    output: str
    reward: float


class SpecialistMemory:
    """
    Per-specialist replay buffer of (task, output, reward) tuples.
    Capped at MAX_PER_SPECIALIST entries; excess low-reward entries are dropped.
    """

    MAX_PER_SPECIALIST = 50

    def __init__(self, path: str = "data/specialist_memory.json"):
        self._path = Path(path)
        self._entries: dict[str, list[MemoryEntry]] = {}
        if self._path.exists():
            self._load()

    def record(
        self,
        specialist_id: str,
        task: str,
        output: str,
        reward: float,
    ) -> None:
        entries = self._entries.setdefault(specialist_id, [])
        entries.append(MemoryEntry(specialist_id, task[:500], output[:800], float(reward)))
        if len(entries) > self.MAX_PER_SPECIALIST:
            entries.sort(key=lambda e: e.reward, reverse=True)
            self._entries[specialist_id] = entries[: self.MAX_PER_SPECIALIST]

    def get_top_examples(self, specialist_id: str, n: int = 5) -> list[MemoryEntry]:
        entries = self._entries.get(specialist_id, [])
        return sorted(entries, key=lambda e: e.reward, reverse=True)[:n]

    def get_failure_examples(self, specialist_id: str, n: int = 3) -> list[MemoryEntry]:
        entries = self._entries.get(specialist_id, [])
        return sorted(entries, key=lambda e: e.reward)[:n]

    def count(self, specialist_id: str) -> int:
        return len(self._entries.get(specialist_id, []))

    def avg_reward(self, specialist_id: str) -> float:
        entries = self._entries.get(specialist_id, [])
        if not entries:
            return 0.0
        return sum(e.reward for e in entries) / len(entries)

    def all_specialist_ids(self) -> list[str]:
        return list(self._entries.keys())

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            sid: [asdict(e) for e in entries]
            for sid, entries in self._entries.items()
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        try:
            with open(self._path) as f:
                data = json.load(f)
            for sid, entries in data.items():
                self._entries[sid] = [MemoryEntry(**e) for e in entries]
        except Exception as exc:
            print(f"[SpecialistMemory] Could not load {self._path}: {exc}")
