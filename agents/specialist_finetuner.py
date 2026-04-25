"""
Specialist Finetuner — evolves specialist system prompts using SpecialistMemory.
Calls GPT-4o-mini with high/low reward examples and asks for an improved prompt.
No-ops gracefully when OPENAI_API_KEY is absent.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.specialist_memory import SpecialistMemory
    from env.specialist_registry import SpecialistRegistry

_MIN_ENTRIES_DEFAULT = 10
_IMPROVE_THRESHOLD_DEFAULT = 0.70   # only improve specialists below this avg reward


class SpecialistFinetuner:
    def __init__(
        self,
        min_entries: int = _MIN_ENTRIES_DEFAULT,
        improve_threshold: float = _IMPROVE_THRESHOLD_DEFAULT,
    ):
        self._min_entries = min_entries
        self._improve_threshold = improve_threshold

    def should_improve(
        self, specialist_id: str, memory: "SpecialistMemory"
    ) -> bool:
        return (
            memory.count(specialist_id) >= self._min_entries
            and memory.avg_reward(specialist_id) < self._improve_threshold
        )

    def improve(
        self,
        specialist_id: str,
        registry: "SpecialistRegistry",
        memory: "SpecialistMemory",
    ) -> bool:
        """
        Generate an improved system prompt via GPT-4o-mini and store it on the
        Specialist object so future _call_openai_specialist calls use it.
        Returns True on success.
        """
        import os
        if not os.getenv("OPENAI_API_KEY"):
            return False

        try:
            specialist = registry.get(specialist_id)
        except KeyError:
            return False

        top = memory.get_top_examples(specialist_id, n=5)
        failed = memory.get_failure_examples(specialist_id, n=3)

        def _fmt(entries):
            if not entries:
                return "(none yet)"
            return "\n".join(
                f"  Task: {e.task[:200]}\n  Output: {e.output[:300]}\n  Reward: {e.reward:.2f}"
                for e in entries
            )

        current_prompt = specialist.system_prompt or "(none — using description only)"
        prompt = (
            f"You are improving the system prompt for a specialist AI agent.\n\n"
            f"Role: {specialist.role}\n"
            f"Description: {specialist.description}\n"
            f"Current system prompt: {current_prompt}\n\n"
            f"HIGH-REWARD examples (keep these patterns):\n{_fmt(top)}\n\n"
            f"LOW-REWARD examples (avoid these patterns):\n{_fmt(failed)}\n\n"
            f"Write an improved system prompt (2–4 sentences) that preserves what "
            f"worked and avoids patterns from low-reward outputs. "
            f"Return ONLY the prompt text, nothing else."
        )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            new_prompt = resp.choices[0].message.content.strip()
            if len(new_prompt) > 30:
                specialist.system_prompt = new_prompt
                print(
                    f"[SpecialistFinetuner] Improved '{specialist_id}' "
                    f"(avg_reward={memory.avg_reward(specialist_id):.2f}, "
                    f"entries={memory.count(specialist_id)})"
                )
                return True
        except Exception as exc:
            print(f"[SpecialistFinetuner] Failed for '{specialist_id}': {exc}")

        return False

    def improve_all(
        self,
        registry: "SpecialistRegistry",
        memory: "SpecialistMemory",
    ) -> int:
        """Run improve() for every eligible specialist. Returns count improved."""
        improved = 0
        for sid in memory.all_specialist_ids():
            if self.should_improve(sid, memory):
                if self.improve(sid, registry, memory):
                    improved += 1
        return improved
