"""
Task Bank — LLM-generated tasks derived from the specialist catalog.

Tasks are generated dynamically using GPT-4o-mini based on:
  1. The sector defined in training_config.yaml
  2. The specialist roster in specialist_catalog.yaml
  3. The current curriculum phase (controls complexity)

No hardcoded task lists. Any sector works by swapping the catalog + sector config.
"""

from __future__ import annotations
import random
import threading
import yaml
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def _load_complexity_config(config_path: str) -> tuple[dict, dict]:
    """Load COMPLEXITY_BY_PHASE and COMPLEXITY_DESCRIPTIONS from config files."""
    import os
    base = os.path.dirname(os.path.abspath(config_path))

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cur = cfg.get("curriculum", {})
    by_phase = {
        1: cur.get("phase1_task_types", ["atomic", "simple"]),
        2: cur.get("phase2_task_types", ["moderate"]),
        3: cur.get("phase3_task_types", ["complex", "enterprise"]),
    }

    desc_path = os.path.join(base, "complexity_descriptions.yaml")
    try:
        with open(desc_path) as f:
            descriptions = yaml.safe_load(f)
    except FileNotFoundError:
        descriptions = {
            "atomic": "a very simple, single-step",
            "simple": "a straightforward, well-scoped",
            "moderate": "a multi-component, realistic",
            "complex": "a complex, multi-system",
            "enterprise": "a large-scale, enterprise-grade",
        }
    return by_phase, descriptions


@dataclass
class Task:
    description: str
    complexity_class: str
    domain: str


class TaskBank:
    """
    Generates tasks dynamically using GPT-4o-mini.
    Falls back to catalog-derived tasks if OpenAI is unavailable.

    Tasks are pre-cached in batches to avoid per-episode API latency.
    """

    def __init__(
        self,
        phase: int = 1,
        config_path: str = "configs/training_config.yaml",
        catalog_path: str = "configs/specialist_catalog.yaml",
    ):
        self.phase = phase
        self._cache: list[Task] = []
        self._client = None
        self._cache_lock = threading.Lock()
        self._refill_running = False

        # Load complexity config from yaml files (not hardcoded)
        self._complexity_by_phase, self._complexity_descriptions = (
            _load_complexity_config(config_path)
        )

        # Load sector config
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        sector_cfg = cfg.get("sector", {})
        self.sector_name = sector_cfg.get("name", "software_engineering")
        self.sector_description = sector_cfg.get(
            "description",
            "Software product development"
        )
        self.use_llm = sector_cfg.get("use_llm_task_generation", True)
        self.llm_model = sector_cfg.get("llm_task_model", "gpt-4o-mini")
        self.cache_size = sector_cfg.get("task_cache_size", 50)

        # Load specialist roles from catalog (for context in prompts)
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f)
        self._specialist_roles = [
            s["role"] for s in catalog.get("specialists", [])
        ]

        if self.use_llm:
            self._init_openai()

        # Pre-fill cache
        self._refill_cache()

    def _init_openai(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            print(f"[TaskBank] OpenAI unavailable: {e}. Using catalog-derived tasks.")
            self._client = None

    def _refill_cache(self):
        """
        Synchronously generate a batch of tasks and extend the cache.
        Thread-safe: holds _cache_lock while writing; clears _refill_running on exit.
        Called directly on first fill (init) and from the background thread thereafter.
        """
        complexities = self._complexity_by_phase.get(self.phase, ["simple"])
        n_per_complexity = max(1, self.cache_size // len(complexities))
        new_tasks: list[Task] = []

        for complexity in complexities:
            if self._client and self.use_llm:
                batch = self._generate_llm_tasks(complexity, n_per_complexity)
            else:
                batch = self._generate_catalog_tasks(complexity, n_per_complexity)
            new_tasks.extend(batch)

        random.shuffle(new_tasks)
        with self._cache_lock:
            self._cache.extend(new_tasks)
            self._refill_running = False

    def _refill_cache_background(self):
        """Trigger a non-blocking background refill if one isn't already running."""
        with self._cache_lock:
            if self._refill_running:
                return          # already in flight — don't pile up threads
            self._refill_running = True

        t = threading.Thread(target=self._refill_cache, daemon=True)
        t.start()

    def _generate_llm_tasks(self, complexity: str, n: int) -> list[Task]:
        """Generate n tasks of the given complexity using GPT-4o-mini.

        Batches requests at max 20 tasks per API call to avoid JSON truncation
        from max_tokens limits. Results are concatenated into a single list.
        """
        complexity_desc = self._complexity_descriptions.get(complexity, "a realistic")
        roles_str = ", ".join(self._specialist_roles)
        batch_size = 20  # safe upper bound — 20 tasks × ~40 tokens each ≈ 800 tokens
        all_tasks: list[Task] = []

        for batch_start in range(0, n, batch_size):
            batch_n = min(batch_size, n - batch_start)
            prompt = f"""You are generating training tasks for a multi-agent RL environment.

Sector: {self.sector_name}
Sector description: {self.sector_description}
Available specialist roles: {roles_str}

Generate exactly {batch_n} different {complexity_desc} task descriptions for this sector.
Each task should:
- Be 1-2 sentences long
- Be specific and realistic for the {self.sector_name} sector
- Potentially require one or more of the available specialists to complete
- Vary in subject matter (don't repeat similar tasks)

Return ONLY a JSON array of strings, no other text:
["task 1 description", "task 2 description", ...]"""

            try:
                import json
                response = self._client.chat.completions.create(
                    model=self.llm_model,
                    max_tokens=1200,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                task_strings = json.loads(raw)
                all_tasks.extend([
                    Task(
                        description=t,
                        complexity_class=complexity,
                        domain=self.sector_name,
                    )
                    for t in task_strings
                    if isinstance(t, str) and len(t) > 10
                ])
            except Exception as e:
                print(f"[TaskBank] LLM generation failed for {complexity} batch: {e}. Using fallback.")
                all_tasks.extend(self._generate_catalog_tasks(complexity, batch_n))

        return all_tasks

    def _generate_catalog_tasks(self, complexity: str, n: int) -> list[Task]:
        """
        Fallback: derive tasks from specialist catalog without API calls.
        Produces formulaic but valid tasks for any sector.
        """
        complexity_desc = self._complexity_descriptions.get(complexity, "a realistic")
        tasks = []
        specialists = self._specialist_roles.copy()
        random.shuffle(specialists)

        for i in range(n):
            if len(specialists) >= 2:
                s1 = specialists[i % len(specialists)]
                s2 = specialists[(i + 1) % len(specialists)]
                desc = (
                    f"Design {complexity_desc} {self.sector_name} solution "
                    f"involving {s1} and {s2} working together"
                )
            else:
                s1 = specialists[0] if specialists else "specialist"
                desc = (
                    f"Create {complexity_desc} {self.sector_name} deliverable "
                    f"for a {s1}"
                )
            tasks.append(Task(
                description=desc,
                complexity_class=complexity,
                domain=self.sector_name,
            ))
        return tasks

    def sample(self) -> str:
        """
        Sample a random task description for a new episode.

        Never blocks for a refill.  When the cache drops below a low-water mark
        (10% of cache_size) a background thread is kicked off to replenish it.
        If the cache is completely empty (should only happen at init or after a
        phase switch drains it before the background fill completes) we fall back
        to a catalog-derived task immediately so reset() is never stalled.
        """
        low_water = max(5, self.cache_size // 10)

        with self._cache_lock:
            if self._cache:
                task = self._cache.pop()
            else:
                task = None

        if task is None:
            # Cache exhausted — generate one catalog task inline (fast, no API)
            fallback = self._generate_catalog_tasks(
                random.choice(self._complexity_by_phase.get(self.phase, ["simple"])), 1
            )
            task_desc = fallback[0].description if fallback else (
                f"Complete a {self.sector_name} task requiring specialist collaboration"
            )
            self._refill_cache_background()
            return task_desc

        with self._cache_lock:
            cache_len = len(self._cache)

        if cache_len < low_water:
            self._refill_cache_background()

        return task.description

    def sample_task(self) -> Task:
        """Sample a full Task object."""
        desc = self.sample()
        complexity = random.choice(self._complexity_by_phase.get(self.phase, ["simple"]))
        return Task(description=desc, complexity_class=complexity, domain=self.sector_name)

    def set_phase(self, phase: int) -> None:
        self.phase = phase
        with self._cache_lock:
            self._cache.clear()
            self._refill_running = False
        self._refill_cache()   # synchronous — phase switches are rare and intentional

    @property
    def pool_size(self) -> int:
        return len(self._cache)
