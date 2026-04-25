"""
Tiered Reward Cascade.

Tiers:
  Tier 0 — Free structural checks (episode completion, no cycles, etc.)
  Tier 1 — Embedding cosine similarity vs task description
  Tier 2 — Small LLM micro-judge (GPT-4o-mini, 3 questions)
  Tier 3 — Full 5-dimension LLM judge (checkpoints only)

Both baseline and specialist output ALWAYS use the same tier (enforced by
EpisodeTierLock). Never subtract across tiers.
"""

from __future__ import annotations
import os
import yaml
import numpy as np
from typing import Optional
from reward.tier_lock import EpisodeTierLock, RewardTier


class TieredRewardScorer:
    """
    Scores outputs at the correct tier for an episode.
    Used to compute: reward_delta = score(specialist_output) - score(baseline)
    """

    def __init__(
        self,
        registry=None,
        rubric_path: str = "configs/reward_rubric.yaml",
    ):
        self._registry = registry
        self._openai_client = None
        self._score_cache: dict[tuple, float] = {}
        try:
            with open(rubric_path) as f:
                self._rubric = yaml.safe_load(f)["tier2_judge"]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"reward_rubric.yaml not found at {rubric_path}. "
                "This file is required — do not delete it."
            )

    def _get_openai_client(self):
        if self._openai_client is None:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            except Exception as e:
                print(f"[Warning] Could not init OpenAI client: {e}")
        return self._openai_client

    def score(
        self,
        output: str,
        task_description: str,
        tier_lock: EpisodeTierLock,
    ) -> float:
        """Score an output at the locked tier. Returns 0.0–1.0.

        Results are cached by (output, task, tier) hash so that scoring the same
        text twice (e.g. generalist baseline scored every episode with the same
        task description, or T2 called for both specialist and baseline) never
        issues a duplicate embedding or LLM call.
        """
        cache_key = (hash(output), hash(task_description), tier_lock.locked_tier.name)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        if tier_lock.locked_tier == RewardTier.TIER_0:
            result = self._tier0_score(output, task_description)
        elif tier_lock.locked_tier == RewardTier.TIER_1:
            result = self._tier1_score(output, task_description)
        elif tier_lock.locked_tier == RewardTier.TIER_2:
            result = self._tier2_score(output, task_description)
        else:
            result = self._tier2_score(output, task_description)  # Tier 3 uses Tier 2 for now

        self._score_cache[cache_key] = result
        return result

    def _tier0_score(self, output: str, task_description: str) -> float:
        """Structural signals: length, non-empty, mentions key task terms."""
        if not output or len(output.strip()) < 20:
            return 0.0

        score = 0.3  # Baseline for non-empty output

        length = len(output)
        if 100 <= length <= 2000:
            score += 0.3
        elif length > 2000:
            score += 0.2
        else:
            score += 0.1

        task_words = set(task_description.lower().split())
        output_words = set(output.lower().split())
        common = task_words & output_words
        overlap = len(common) / max(len(task_words), 1)
        score += min(overlap * 0.4, 0.4)

        return min(score, 1.0)

    def _tier1_score(self, output: str, task_description: str) -> float:
        """Embedding cosine similarity between output and task."""
        if self._registry is None:
            return self._tier0_score(output, task_description)

        try:
            task_emb = self._registry.embed_query(task_description)
            output_emb = self._registry.embed_query(output[:1000])
            similarity = self._registry.cosine_similarity(task_emb, output_emb)
            # Map from [-1, 1] cosine similarity to [0, 1] reward range
            return float((similarity + 1.0) / 2.0)
        except Exception:
            return self._tier0_score(output, task_description)

    def _tier2_score(self, output: str, task_description: str) -> float:
        """
        Small LLM micro-judge. Rubric dimensions, model, and normalisation
        denominator are read from configs/reward_rubric.yaml — not hardcoded.
        Returns 0.0–1.0.
        """
        client = self._get_openai_client()
        if client is None:
            return self._tier1_score(output, task_description)

        dims = self._rubric["dimensions"]
        model = self._rubric.get("model", "gpt-4o-mini")
        max_tokens = self._rubric.get("max_tokens", 100)
        denom = self._rubric.get("normalisation_denominator", 11)

        dim_lines = "\n".join(
            f"- {k}: {v['scale']}"
            for k, v in dims.items()
        )
        json_template = ", ".join(
            f'"{k}": <{v["min"]}-{v["max"]}>'
            for k, v in dims.items()
        )
        prompt = (
            f"You are evaluating an AI assistant's output. "
            f"Answer {len(dims)} questions:\n\n"
            f"Task: {task_description[:500]}\n\n"
            f"Output: {output[:800]}\n\n"
            f"Answer ONLY with this JSON format, nothing else:\n"
            f"{{{json_template}}}\n\n"
            f"{dim_lines}"
        )

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            import json
            text = response.choices[0].message.content.strip()
            scores = json.loads(text)
            required_keys = set(dims.keys())
            if not required_keys.issubset(scores):
                missing = required_keys - scores.keys()
                print(f"[Tier2Judge] Missing keys {missing} in response: {text}. Falling back.")
                return self._tier1_score(output, task_description)
            total = sum(
                max(v["min"], min(v["max"], int(scores[k])))
                for k, v in dims.items()
            )
            return float(total) / float(denom)
        except Exception as e:
            print(f"[Tier2Judge] Error: {e}. Falling back to Tier 1.")
            return self._tier1_score(output, task_description)
