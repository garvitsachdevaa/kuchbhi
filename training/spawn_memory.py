"""
SpawnMemory — tracks which specialist descriptions worked for which tasks.

Used to condition future spawn prompts on past successes.
This is retrieval-augmented generation for specialist design.
Path is configurable via environment.spawn_memory_path in training_config.yaml.
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SpawnRecord:
    task_embedding:   list[float]   # 384-dim stored as list for JSON serialisation
    task_description: str
    specialist_id:    str
    specialist_role:  str
    specialist_desc:  str
    episode_reward:   float         # terminal reward of the episode that triggered the spawn
    pre_spawn_sim:    float
    post_spawn_sim:   float
    episode_idx:      int


class SpawnMemory:
    """
    File-backed JSONL memory of past spawns with cosine-similarity retrieval.
    Capped at max_entries; lowest-reward records are evicted when full.
    """

    def __init__(self, path: str, max_entries: int = 500):
        self._path = Path(path)
        self.max_entries = max_entries
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[SpawnRecord] = self._load()

    def _load(self) -> list[SpawnRecord]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text().splitlines():
            try:
                records.append(SpawnRecord(**json.loads(line)))
            except Exception:
                continue
        return records

    def record(self, rec: SpawnRecord) -> None:
        self._records.append(rec)
        if len(self._records) > self.max_entries:
            self._records.sort(key=lambda r: r.episode_reward, reverse=True)
            self._records = self._records[: self.max_entries]
        with open(self._path, "w") as f:
            for r in self._records:
                f.write(json.dumps(asdict(r)) + "\n")

    def retrieve_similar(
        self,
        task_embedding: np.ndarray,
        top_k: int = 3,
        min_reward: float = 0.0,
    ) -> list[SpawnRecord]:
        """
        Return top_k past spawns whose task was most similar to the current
        task, filtered to those that produced >= min_reward.
        """
        if not self._records:
            return []
        candidates = [r for r in self._records if r.episode_reward >= min_reward]
        if not candidates:
            return []
        norm_task = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)
        scored = []
        for rec in candidates:
            emb = np.array(rec.task_embedding, dtype=np.float32)
            norm_emb = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(norm_emb, norm_task))
            scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    @property
    def size(self) -> int:
        return len(self._records)
