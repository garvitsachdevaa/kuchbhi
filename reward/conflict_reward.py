"""
Conflict detection and resolution reward signals.

Detection strategy:
  1. Primary: Embedding-similarity contradiction detection
     Two outputs are in conflict if they are semantically dissimilar
     despite addressing the same task (cosine sim < threshold).
  2. Fallback: Keyword-based detection using sector-defined contradiction
     pairs loaded from the specialist catalog (optional field).

No domain-specific logic is hardcoded here.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class ConflictType(Enum):
    FACTUAL   = "factual"
    TECHNICAL = "technical"
    PRIORITY  = "priority"
    SCOPE     = "scope"


@dataclass
class Conflict:
    conflict_type: ConflictType
    agent_a: str
    agent_b: str
    description: str
    resolved: bool = False


def detect_conflicts(
    results,
    registry=None,
    contradiction_pairs: Optional[list[tuple[str, str]]] = None,
    similarity_threshold: float = 0.25,
) -> list[Conflict]:
    """
    Detect conflicts between specialist outputs.

    Two detection methods, tried in order:
    1. Embedding similarity (if registry provided): outputs covering the same
       task that are semantically distant from each other are flagged as
       conflicting. Threshold: cosine similarity < similarity_threshold.
    2. Keyword contradiction pairs (if provided via catalog or caller):
       domain-specific term pairs that signal contradiction.

    Args:
        results: List of SpecialistResult objects
        registry: SpecialistRegistry instance (for embedding-based detection)
        contradiction_pairs: Optional list of (term_a, term_b) tuples loaded
                             from the sector's specialist catalog
        similarity_threshold: Cosine similarity below which outputs are flagged
    """
    conflicts = []
    outputs = [
        (r.specialist_id, r.output)
        for r in results
        if r.output and len(r.output.strip()) > 20
    ]

    if len(outputs) < 2:
        return conflicts

    # Method 1: Embedding-based conflict detection
    if registry is not None:
        embedding_conflicts = _detect_embedding_conflicts(
            outputs, registry, similarity_threshold
        )
        conflicts.extend(embedding_conflicts)

    # Method 2: Keyword-based (sector-defined pairs, not hardcoded)
    if contradiction_pairs:
        keyword_conflicts = _detect_keyword_conflicts(outputs, contradiction_pairs)
        # Deduplicate against already-found conflicts
        existing = {(c.agent_a, c.agent_b) for c in conflicts}
        for c in keyword_conflicts:
            if (c.agent_a, c.agent_b) not in existing:
                conflicts.append(c)

    return conflicts


def _detect_embedding_conflicts(
    outputs: list[tuple[str, str]],
    registry,
    threshold: float,
) -> list[Conflict]:
    """
    Flag pairs of outputs that are semantically distant as potential conflicts.
    Uses cosine similarity on output embeddings.
    """
    conflicts = []
    embeddings = {}

    for agent_id, output in outputs:
        try:
            emb = registry.embed_query(output[:500])
            embeddings[agent_id] = emb
        except Exception:
            continue

    agent_ids = list(embeddings.keys())
    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            id_a = agent_ids[i]
            id_b = agent_ids[j]
            sim = registry.cosine_similarity(embeddings[id_a], embeddings[id_b])
            if sim < threshold:
                conflicts.append(Conflict(
                    conflict_type=ConflictType.TECHNICAL,
                    agent_a=id_a,
                    agent_b=id_b,
                    description=(
                        f"Semantic divergence between {id_a} and {id_b} "
                        f"(cosine similarity: {sim:.3f} < {threshold})"
                    ),
                    resolved=False,
                ))

    return conflicts


def _detect_keyword_conflicts(
    outputs: list[tuple[str, str]],
    contradiction_pairs: list[tuple[str, str]],
) -> list[Conflict]:
    """
    Keyword-based conflict detection using sector-provided contradiction pairs.
    These pairs are loaded from specialist_catalog.yaml, NOT hardcoded here.
    """
    conflicts = []
    for i, (id_a, out_a) in enumerate(outputs):
        for id_b, out_b in outputs[i + 1:]:
            out_a_lower = out_a.lower()
            out_b_lower = out_b.lower()
            for term_a, term_b in contradiction_pairs:
                if (
                    (term_a in out_a_lower and term_b in out_b_lower) or
                    (term_b in out_a_lower and term_a in out_b_lower)
                ):
                    conflicts.append(Conflict(
                        conflict_type=ConflictType.TECHNICAL,
                        agent_a=id_a,
                        agent_b=id_b,
                        description=f"Keyword contradiction: {term_a} vs {term_b}",
                        resolved=False,
                    ))
    return conflicts
