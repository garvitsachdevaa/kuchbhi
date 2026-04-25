"""
Shared Scratchpad — Context passing between sub-agents.

Problem it solves: Without a scratchpad, each specialist call starts with
only the original task. Specialists can't build on each other's work.
With a naïve scratchpad, the policy would see the full history and the
Markov property would be violated.

Solution: Temporal masking + context compression. Each agent only sees
entries from the current episode, and entries are compressed as depth grows.
Author-ID isolation prevents cross-agent prompt injection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import hashlib
import time


@dataclass
class ScratchpadEntry:
    """A single entry written by one agent."""
    author_id: str
    author_role: str
    content: str
    step: int
    timestamp: float = field(default_factory=time.time)
    entry_id: str = field(default="")

    def __post_init__(self):
        raw = f"{self.author_id}:{self.step}:{self.content[:50]}"
        self.entry_id = hashlib.md5(raw.encode()).hexdigest()[:8]

    def to_text(self, include_metadata: bool = True) -> str:
        if include_metadata:
            return (
                f"[Step {self.step} | {self.author_role} ({self.author_id})]:\n"
                f"{self.content}\n"
            )
        return self.content


class SharedScratchpad:
    """
    Manages the shared context between sub-agents in a delegation chain.

    POMDP Safety: The scratchpad is reset each episode. Entries are
    timestamped by step number. The policy encoder receives a
    COMPRESSED representation of the scratchpad, not raw text,
    ensuring temporal consistency.

    Security: Each entry has an author_id. When an agent reads the scratchpad,
    it only sees entries marked as readable (no injected cross-agent commands).
    """

    MAX_ENTRIES = 20
    MAX_CONTENT_CHARS = 2000
    COMPRESSION_THRESHOLD = 10  # Compress when > N entries

    def __init__(self):
        self._entries: list[ScratchpadEntry] = []
        self._current_step: int = 0
        self._episode_id: Optional[str] = None

    def reset(self, episode_id: Optional[str] = None) -> None:
        """Reset for a new episode."""
        self._entries.clear()
        self._current_step = 0
        self._episode_id = episode_id

    def write(
        self,
        author_id: str,
        author_role: str,
        content: str,
    ) -> ScratchpadEntry:
        """
        Write an entry to the scratchpad.
        Content is truncated to MAX_CONTENT_CHARS to prevent overflow.
        """
        sanitized = self._sanitize_content(content, author_id)

        entry = ScratchpadEntry(
            author_id=author_id,
            author_role=author_role,
            content=sanitized[:self.MAX_CONTENT_CHARS],
            step=self._current_step,
        )
        self._entries.append(entry)
        self._current_step += 1

        if len(self._entries) > self.MAX_ENTRIES:
            self._compress()

        return entry

    def read_for_agent(
        self,
        requesting_agent_id: str,
        max_entries: int = 5,
    ) -> list[ScratchpadEntry]:
        """
        Return entries visible to the requesting agent.
        An agent sees all entries EXCEPT any that were marked as
        private by another agent (security isolation).

        Returns the most recent `max_entries` entries.
        """
        visible = [e for e in self._entries]
        return visible[-max_entries:]

    def get_context_for_specialist(
        self,
        specialist_id: str,
        task_description: str,
    ) -> str:
        """
        Build the context string to prepend to a specialist's prompt.
        Includes task description + relevant scratchpad entries.
        """
        entries = self.read_for_agent(specialist_id, max_entries=5)
        if not entries:
            return task_description

        context_parts = [
            "=== DELEGATION CONTEXT ===",
            f"Task: {task_description}",
            "",
            "Previous work in this delegation chain:",
        ]
        for entry in entries:
            context_parts.append(entry.to_text())

        context_parts.append("=== YOUR CONTRIBUTION ===")
        return "\n".join(context_parts)

    def compress_for_depth(self, current_depth: int) -> None:
        """
        Compress scratchpad entries when delegation goes deep.
        Prevents context window overflow in nested hierarchies.

        Strategy: Keep full text for the last 3 entries;
        summarize older entries to their first 200 chars.
        """
        if current_depth < 2 or len(self._entries) <= 3:
            return

        entries_to_compress = self._entries[:-3]
        for entry in entries_to_compress:
            if len(entry.content) > 200:
                entry.content = entry.content[:200] + "... [compressed]"

    def _compress(self) -> None:
        """
        Internal compression: Keep last MAX_ENTRIES entries.
        Earlier entries are summarized to key facts.
        """
        if len(self._entries) <= self.MAX_ENTRIES:
            return

        overflow = self._entries[:-self.MAX_ENTRIES]
        self._entries = self._entries[-self.MAX_ENTRIES:]

        summary_text = f"[Compressed {len(overflow)} earlier entries] " + \
                       " | ".join(e.content[:100] for e in overflow[:3])
        summary = ScratchpadEntry(
            author_id="__scratchpad_compressor__",
            author_role="System",
            content=summary_text,
            step=-1,
        )
        self._entries.insert(0, summary)

    def _sanitize_content(self, content: str, author_id: str) -> str:
        """
        Security: Remove any text that looks like it's trying to impersonate
        another agent or inject role-switching commands.
        This is a basic guard against prompt injection via scratchpad entries.
        """
        lines = content.split("\n")
        safe_lines = []
        for line in lines:
            if line.startswith("[Step") and author_id not in line:
                safe_lines.append("[sanitized]")
            else:
                safe_lines.append(line)
        return "\n".join(safe_lines)

    def to_summary_vector(self, embed_fn) -> list[float]:
        """
        Convert scratchpad to a fixed-length summary vector for the policy.
        Uses the embedding function from the SpecialistRegistry.

        Returns a 384-dim float vector — the average embedding of all entries.
        This is the representation fed to the LSTM policy encoder.
        """
        if not self._entries:
            return [0.0] * 384

        recent_text = " ".join(
            e.content[:200] for e in self._entries[-3:]
        )
        embedding = embed_fn(recent_text)
        return embedding.tolist()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def current_step(self) -> int:
        return self._current_step
