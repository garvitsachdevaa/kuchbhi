"""
Specialist Registry — Dynamic roster with capability embeddings.

Design principle: The policy operates on capability embedding vectors,
not specialist IDs. The YAML catalog is a BOOTSTRAP SEED only — not a
closed enum. New specialists can be added at any time via add_specialist()
and the policy represents them immediately through their embedding.

This is the core property that separates this from a classifier:
- Classifier: breaks when you add a new specialist (unseen class)
- This registry: new specialists are immediately representable zero-shot
"""

from __future__ import annotations
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers import SentenceTransformer


@dataclass
class Specialist:
    """
    Represents a single specialist agent in the roster.
    The embedding is computed once at registry init and cached.
    """
    id: str
    role: str
    description: str
    complexity_affinity: list[str]
    avg_latency_ms: float
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    system_prompt: Optional[str] = field(default=None, repr=False)

    def to_state_vector(self) -> np.ndarray:
        """Return the embedding vector for use in state representation."""
        if self.embedding is None:
            raise RuntimeError(f"Specialist {self.id} embedding not computed yet.")
        return self.embedding.astype(np.float32)


class SpecialistRegistry:
    """
    Manages the available specialist roster.

    Key design decisions:
    - Uses all-MiniLM-L6-v2 (384-dim, local, free, no API calls)
    - Embeddings computed once at init, cached in memory
    - Supports dynamic addition of new specialists without breaking policy
    - State representation is always 384-dim per specialist (roster-agnostic)
    """

    EMBEDDING_DIM = 384
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, catalog_path: str | Path, lazy_load: bool = False):
        self.catalog_path = Path(catalog_path)
        self._model: Optional[SentenceTransformer] = None
        self._specialists: dict[str, Specialist] = {}

        with open(self.catalog_path, "r") as f:
            catalog = yaml.safe_load(f)

        for spec_data in catalog["specialists"]:
            specialist = Specialist(
                id=spec_data["id"],
                role=spec_data["role"],
                description=spec_data["description"],
                complexity_affinity=spec_data["complexity_affinity"],
                avg_latency_ms=spec_data["avg_latency_ms"],
            )
            self._specialists[specialist.id] = specialist

        if not lazy_load:
            self._load_model_and_embed()

    def _load_model_and_embed(self) -> None:
        """Load sentence transformer and compute all embeddings."""
        print(f"[SpecialistRegistry] Loading embedding model: {self.MODEL_NAME}")
        self._model = SentenceTransformer(self.MODEL_NAME)

        descriptions = [s.description for s in self._specialists.values()]
        embeddings = self._model.encode(descriptions, normalize_embeddings=True)

        for specialist, embedding in zip(self._specialists.values(), embeddings):
            specialist.embedding = embedding.astype(np.float32)

        print(f"[SpecialistRegistry] Embedded {len(self._specialists)} specialists "
              f"(dim={self.EMBEDDING_DIM})")

    def get(self, specialist_id: str) -> Specialist:
        if specialist_id not in self._specialists:
            raise KeyError(f"Unknown specialist: {specialist_id}")
        return self._specialists[specialist_id]

    def list_ids(self) -> list[str]:
        return list(self._specialists.keys())

    def list_all(self) -> list[Specialist]:
        return list(self._specialists.values())

    @property
    def size(self) -> int:
        return len(self._specialists)

    def get_embeddings_matrix(self) -> np.ndarray:
        """
        Returns shape (N, 384) matrix of all specialist embeddings.
        Used by the policy encoder to compute attention over the roster.
        """
        return np.stack([s.to_state_vector() for s in self._specialists.values()])

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed an arbitrary text query (e.g., task description).
        Used for similarity-based matching and Tier 1 reward.
        """
        if self._model is None:
            self._load_model_and_embed()
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

    def add_specialist(self, specialist_data: dict) -> None:
        """
        Dynamically add a new specialist to the roster.
        Policy can immediately represent it via its embedding.
        This is called BETWEEN training runs (not during episodes),
        consistent with the SPAWN_SPECIALIST meta-level design.
        """
        specialist = Specialist(
            id=specialist_data["id"],
            role=specialist_data["role"],
            description=specialist_data["description"],
            complexity_affinity=specialist_data["complexity_affinity"],
            avg_latency_ms=specialist_data["avg_latency_ms"],
        )
        if self._model is not None:
            embedding = self._model.encode(
                specialist.description, normalize_embeddings=True
            )
            specialist.embedding = embedding.astype(np.float32)
        self._specialists[specialist.id] = specialist
        print(f"[SpecialistRegistry] Added specialist: {specialist.id}")

    def get_specialists_for_complexity(
        self, complexity_class: str
    ) -> list[Specialist]:
        """Return specialists appropriate for a given task complexity."""
        return [
            s for s in self._specialists.values()
            if complexity_class in s.complexity_affinity
        ]

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def find_most_similar(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """
        Find the top-k specialists most similar to a query embedding.
        Returns list of (specialist_id, similarity_score) tuples.
        """
        similarities = []
        for specialist in self._specialists.values():
            sim = self.cosine_similarity(query_embedding, specialist.to_state_vector())
            similarities.append((specialist.id, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
