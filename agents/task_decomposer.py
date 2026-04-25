"""
Task Decomposer — handles task ambiguity before episode starts.
Two modes: INTERACTIVE (asks for clarification) and AUTONOMOUS (infers defaults).
For hackathon: uses AUTONOMOUS mode (95% of enterprise use cases).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import os
import yaml


class ComplexityClass(Enum):
    ATOMIC     = "atomic"
    SIMPLE     = "simple"
    MODERATE   = "moderate"
    COMPLEX    = "complex"
    ENTERPRISE = "enterprise"


def _load_complexity_keywords(
    keywords_path: str = "configs/complexity_keywords.yaml",
) -> dict[str, list[str]]:
    try:
        with open(keywords_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"complexity_keywords.yaml not found at {keywords_path}. "
            "This file is required — do not delete it."
        )


@dataclass
class EnrichedTask:
    """Task with inferred metadata for episode setup."""
    original_description: str
    enriched_description: str
    complexity_class: str
    expected_specialists: int
    domain_hints: list[str]
    is_ambiguous: bool
    autonomously_enriched: bool


class TaskDecomposer:
    """
    Analyzes task descriptions and enriches them with inferred metadata.
    Fully implemented — no 'pass' stubs.
    """

    DOMAIN_KEYWORDS = {
        "frontend":  ["react", "vue", "angular", "ui", "css", "frontend", "component"],
        "backend":   ["api", "server", "endpoint", "rest", "backend", "node", "express"],
        "database":  ["database", "schema", "sql", "mongodb", "postgresql", "redis"],
        "devops":    ["deploy", "docker", "kubernetes", "ci/cd", "pipeline", "cloud"],
        "security":  ["auth", "security", "encryption", "oauth", "jwt", "compliance"],
        "product":   ["requirement", "feature", "user story", "roadmap", "mvp"],
    }

    COMPLEXITY_SPECIALIST_MAP = {
        "atomic":     1,
        "simple":     2,
        "moderate":   3,
        "complex":    4,
        "enterprise": 5,
    }

    def __init__(
        self,
        sector_cfg: dict | None = None,
        keywords_path: str = "configs/complexity_keywords.yaml",
    ):
        # sector.default_assumptions is required — no silent React/Node fallback
        assumptions = (sector_cfg or {}).get("default_assumptions")
        if assumptions is None:
            raise ValueError(
                "sector.default_assumptions is missing from training_config.yaml. "
                "Add frontend/backend/database/team_size keys under sector.default_assumptions."
            )
        self._assumptions = assumptions
        self._complexity_keywords = _load_complexity_keywords(keywords_path)

    def decompose(self, task_description: str) -> EnrichedTask:
        """Main entry point. Returns an EnrichedTask."""
        complexity = self._classify_complexity(task_description)
        domains = self._detect_domains(task_description)
        is_ambiguous = self._is_ambiguous(task_description)

        enriched_desc = self.enrich_with_defaults(
            task_description, complexity, domains, is_ambiguous
        )

        return EnrichedTask(
            original_description=task_description,
            enriched_description=enriched_desc,
            complexity_class=complexity,
            expected_specialists=self.COMPLEXITY_SPECIALIST_MAP[complexity],
            domain_hints=domains,
            is_ambiguous=is_ambiguous,
            autonomously_enriched=is_ambiguous,
        )

    def _classify_complexity(self, description: str) -> str:
        desc_lower = description.lower()
        for complexity in ["enterprise", "complex", "moderate", "simple", "atomic"]:
            keywords = self._complexity_keywords.get(complexity, [])
            if any(kw in desc_lower for kw in keywords):
                return complexity
        word_count = len(description.split())
        if word_count > 15:
            return "moderate"
        elif word_count > 8:
            return "simple"
        else:
            return "atomic"

    def _detect_domains(self, description: str) -> list[str]:
        desc_lower = description.lower()
        detected = []
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                detected.append(domain)
        return detected if detected else ["general"]

    def _is_ambiguous(self, description: str) -> bool:
        if len(description.split()) < 4:
            return True
        vague_words = ["it", "this", "that", "something", "stuff", "thing"]
        desc_lower = description.lower()
        vague_count = sum(1 for w in vague_words if f" {w} " in f" {desc_lower} ")
        return vague_count >= 2

    def enrich_with_defaults(
        self,
        description: str,
        complexity: str,
        domains: list[str],
        is_ambiguous: bool,
    ) -> str:
        """
        Enrich ambiguous tasks with sector-configured technology assumptions.
        Reads from self._assumptions (sector.default_assumptions in config).
        """
        if not is_ambiguous:
            return description

        enriched = description
        desc_lower = description.lower()

        frontend_stack = self._assumptions.get("frontend", "")
        backend_stack  = self._assumptions.get("backend", "")
        database_stack = self._assumptions.get("database", "")
        team_size      = self._assumptions.get("team_size", "")

        if "frontend" in domains and frontend_stack:
            if not any(w in desc_lower for w in frontend_stack.lower().split("/")):
                enriched += f" (assume {frontend_stack} frontend)"

        if "backend" in domains and backend_stack:
            if not any(w in desc_lower for w in backend_stack.lower().split("/")):
                enriched += f" (assume {backend_stack} backend)"

        if "database" in domains and database_stack:
            if not any(w in desc_lower for w in database_stack.lower().split("/")):
                enriched += f" (assume {database_stack} database)"

        if complexity in ["moderate", "complex"] and team_size and "scale" not in desc_lower:
            enriched += f" for a team of {team_size}"

        return enriched
