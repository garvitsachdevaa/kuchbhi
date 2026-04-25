"""
Fallback chain resolver — handles specialist failures with graceful degradation.

Fallback chains are loaded from the specialist catalog (optional field).
If not defined in the catalog, a default strategy is used:
  - Try any specialist that shares a complexity_affinity with the failed one
  - Fall back to the lowest-latency specialist as last resort
"""

from __future__ import annotations
import yaml
from pathlib import Path
from reward.failure_reward import SpecialistResult, SpecialistStatus


class FallbackChainResolver:
    """
    If a specialist fails, automatically selects a fallback specialist.
    Chains are loaded from the catalog; no hardcoded specialist IDs.
    """

    def __init__(self, catalog_path: str = "configs/specialist_catalog.yaml"):
        self._chains: dict[str, list[str]] = {}
        self._specialists: list[dict] = []
        self._load_catalog(catalog_path)

    def _load_catalog(self, catalog_path: str) -> None:
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f)

        self._specialists = catalog.get("specialists", [])

        # Load explicit fallback chains if defined in catalog
        for spec in self._specialists:
            if "fallback_to" in spec:
                self._chains[spec["id"]] = spec["fallback_to"]

    def get_fallback(
        self, failed_specialist_id: str, already_called: list[str]
    ) -> str | None:
        """
        Return the next fallback specialist, or None if exhausted.

        Priority:
        1. Explicit fallback_to chain from catalog
        2. Specialist sharing complexity_affinity with the failed one
        3. Lowest-latency available specialist
        """
        # 1. Explicit chain
        if failed_specialist_id in self._chains:
            for fallback_id in self._chains[failed_specialist_id]:
                if fallback_id not in already_called:
                    return fallback_id

        # 2. Shared complexity affinity
        failed_spec = next(
            (s for s in self._specialists if s["id"] == failed_specialist_id), None
        )
        if failed_spec:
            failed_affinities = set(failed_spec.get("complexity_affinity", []))
            candidates = [
                s for s in self._specialists
                if s["id"] != failed_specialist_id
                and s["id"] not in already_called
                and set(s.get("complexity_affinity", [])) & failed_affinities
            ]
            if candidates:
                # Pick lowest latency among affinity-compatible specialists
                candidates.sort(key=lambda s: s.get("avg_latency_ms", 9999))
                return candidates[0]["id"]

        # 3. Any available specialist (lowest latency)
        available = [
            s for s in self._specialists
            if s["id"] != failed_specialist_id
            and s["id"] not in already_called
        ]
        if available:
            available.sort(key=lambda s: s.get("avg_latency_ms", 9999))
            return available[0]["id"]

        return None

    def needs_fallback(self, result: SpecialistResult) -> bool:
        return result.status in (
            SpecialistStatus.TIMEOUT,
            SpecialistStatus.ERROR,
        )
