"""
SpindleFlowEnv — Main RL environment.
Gymnasium-compatible. Wraps SpindleFlow as the execution backend.
LSTM-policy-safe: state representation is complete per-step (no hidden history).

The environment does NOT call SpindleFlow for every episode during training —
that would be too slow and expensive. Instead, for Phase 1/2 training it uses
a simulated specialist execution (fast, free). For evaluation and demo, it
calls real SpindleFlow.
"""

from __future__ import annotations
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Optional, Any
import yaml

from env.specialist_registry import SpecialistRegistry
from env.delegation_graph import DelegationGraph
from env.scratchpad import SharedScratchpad
from env.state import build_state, EpisodeState
from env.action_space import ActionDecoder, MetaAction, FactoredAction, DelegationMode
from reward.tier_lock import EpisodeTierLock
from reward.tiered_reward import TieredRewardScorer
from reward.latency_reward import LatencySLAConfig, compute_latency_penalty
from reward.failure_reward import (
    SpecialistResult, SpecialistStatus,
    compute_failure_penalty, compute_recovery_bonus,
)
from reward.conflict_reward import detect_conflicts
from reward.consistency_tracker import PathConsistencyTracker
from agents.task_decomposer import TaskDecomposer, EnrichedTask
from agents.conflict_resolver import ConflictResolver
from agents.fallback_chain import FallbackChainResolver
from agents.specialist_memory import SpecialistMemory
from training.spawn_memory import SpawnMemory, SpawnRecord
from training.task_bank import TaskBank


class SpindleFlowEnv(gym.Env):
    """
    RL Environment for SpindleFlow delegation policy training.

    Episode structure:
      1. Reset: Draw task from task bank, embed it, lock tier, set up components
      2. Step loop: Policy chooses action → environment executes → compute reward
      3. Termination: STOP action, max_steps reached, or episode error

    Observation space: Flat vector (see EpisodeState.observation_dim())
    Action space: Box (continuous — decoded by ActionDecoder)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        catalog_path: str = "configs/specialist_catalog.yaml",
        use_real_spindleflow: bool = False,
        phase: int = 1,
        render_mode: Optional[str] = None,
        simulate_specialists: bool = False,
    ):
        super().__init__()

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        env_cfg = self.config["environment"]
        self.max_steps = env_cfg["max_steps_per_episode"]
        self.max_depth = env_cfg["max_delegation_depth"]
        self.max_specialists = env_cfg.get("max_specialists_per_episode", 6)
        self.specialist_timeout_ms = env_cfg["specialist_timeout_ms"]
        self.phase = phase
        self.use_real_spindleflow = use_real_spindleflow
        self.render_mode = render_mode
        # When True: per-step specialist calls use simulation even if OPENAI_API_KEY
        # is set. Episode-level self-learning (finetuner, spawn) still use the key.
        self.simulate_specialists = simulate_specialists

        reward_cfg = self.config["reward"]
        self.latency_sla = LatencySLAConfig(
            budget_ms=10000.0,
            weight=reward_cfg["latency_weight"],
        )

        # Initialize components
        self.registry = SpecialistRegistry(catalog_path)
        self.task_bank = TaskBank(
            phase=phase,
            config_path=config_path,
            catalog_path=catalog_path,
        )
        # Load sector contradiction pairs from catalog (for conflict detection)
        with open(catalog_path) as _f:
            _catalog_meta = yaml.safe_load(_f).get("metadata", {})
        self._contradiction_pairs = [
            tuple(pair) for pair in _catalog_meta.get("contradiction_pairs", [])
        ]

        self.task_decomposer = TaskDecomposer(sector_cfg=self.config.get("sector", {}))
        _resolution_mem_path = self.config.get("agents", {}).get(
            "resolution_memory_path", "data/resolution_memory.jsonl"
        )
        self.conflict_resolver = ConflictResolver(
            config=self.config,
            memory_path=_resolution_mem_path,
        )
        self.fallback_resolver = FallbackChainResolver()
        self.reward_scorer = TieredRewardScorer(registry=self.registry)
        self.consistency_tracker = PathConsistencyTracker(
            specialist_ids=self.registry.list_ids()
        )
        si_cfg = self.config.get("specialist_improvement", {})
        memory_path = si_cfg.get("memory_path", "data/specialist_memory.json")
        self.specialist_memory = SpecialistMemory(path=memory_path)

        spawn_mem_path = env_cfg.get("spawn_memory_path", "data/spawn_memory.jsonl")
        self._spawn_memory = SpawnMemory(
            path=spawn_mem_path,
            max_entries=env_cfg.get("spawn_memory_max_entries", 500),
        )
        self._pending_spawn_records: list[SpawnRecord] = []
        self.action_decoder = ActionDecoder(
            specialist_ids=self.registry.list_ids(),
            max_specialists=self.max_specialists,
        )

        # Spawn config
        self.spawn_threshold: float = env_cfg.get("spawn_threshold", 0.50)
        self.auto_spawn: bool = env_cfg.get("auto_spawn_specialists", True)
        # Max total spawned specialists across the lifetime of this env instance.
        # Caps registry growth so the observation space stays stable during long runs.
        self._spawn_max_total: int = env_cfg.get("spawn_max_total", 8)
        # Minimum episodes between consecutive spawns — prevents burst-spawning on
        # a streak of low-similarity tasks and keeps the action decoder stable.
        self._spawn_cooldown_episodes: int = env_cfg.get("spawn_cooldown_episodes", 20)
        # Lifetime counters (survive across resets)
        self._spawn_total_count: int = 0
        self._last_spawn_episode: int = -999  # episode index of last spawn
        self._episode_index: int = 0

        # Per-episode state
        self.delegation_graph = DelegationGraph(max_depth=self.max_depth)
        self.scratchpad = SharedScratchpad()
        self.current_task: Optional[EnrichedTask] = None
        self.tier_lock: Optional[EpisodeTierLock] = None
        self.specialist_results: list[SpecialistResult] = []
        self.called_ids: list[str] = []
        self.step_count: int = 0
        self.episode_start_ms: float = 0.0
        self.generalist_baseline: str = ""
        self.config_reward = reward_cfg
        self._last_reward_components: dict = {}
        self._last_factored_action: Optional[Any] = None
        # Active roster for this episode (top-K by task similarity, including spawned)
        self.active_specialist_ids: list[str] = self.registry.list_ids()[:self.max_specialists]
        self.spawned_this_episode: list[str] = []
        # Task embedding cached at reset() — constant within an episode, no need to re-embed each step
        self._task_emb: np.ndarray | None = None

        # Spaces
        obs_dim = EpisodeState.observation_dim(self.max_specialists)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_decoder.get_action_dim(),),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.delegation_graph.reset()
        self.scratchpad.reset(episode_id=str(time.time()))
        self.specialist_results = []
        self.called_ids = []
        self.step_count = 0
        self.episode_start_ms = time.time() * 1000

        task_desc = self.task_bank.sample()
        self.current_task = self.task_decomposer.decompose(task_desc)

        self.tier_lock = EpisodeTierLock.for_task(
            self.current_task.complexity_class
        )

        self.generalist_baseline = self._generate_generalist_baseline(
            self.current_task.enriched_description
        )

        self.delegation_graph.add_root("orchestrator")
        self._episode_index += 1

        task_desc = self.current_task.enriched_description
        task_emb  = self.registry.embed_query(task_desc)
        assert task_emb is not None and task_emb.shape == (384,), (
            f"Task embedding failed: got shape {getattr(task_emb, 'shape', None)}"
        )
        self._task_emb = task_emb   # cached for entire episode — task doesn't change

        self.spawned_this_episode = []
        self._pending_spawn_records = []
        # Auto-spawn: if no existing specialist covers this task well, create one via LLM.
        if self.auto_spawn:
            self._maybe_spawn_specialist(task_emb, task_desc)

        # ── Build per-episode active roster (top-K by task similarity) ──
        self.active_specialist_ids = self._select_active_specialists(task_emb)

        # ── Rebuild action decoder to reflect the updated roster ──
        self.action_decoder = ActionDecoder(
            specialist_ids=self.active_specialist_ids,
            max_specialists=self.max_specialists,
        )

        state = build_state(
            task_embedding=task_emb,
            registry=self.registry,
            called_ids=[],
            delegation_graph=self.delegation_graph,
            scratchpad=self.scratchpad,
            step_count=0,
            elapsed_ms=0.0,
            sla_budget_ms=self.latency_sla.budget_ms,
            max_specialists=self.max_specialists,
            max_depth=self.max_depth,
            phase=self.phase,
            active_ids=self.active_specialist_ids,
        )

        info = {
            "task":               task_desc,
            "complexity":         self.current_task.complexity_class,
            "tier":               self.tier_lock.locked_tier.name,
            "active_specialists": list(self.active_specialist_ids),
            "spawned_specialists": list(self.spawned_this_episode),
        }

        return state.to_flat_vector(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        Returns: (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        elapsed_ms = time.time() * 1000 - self.episode_start_ms

        # Build specialist mask (enforce DAG constraints)
        valid_ids = self.delegation_graph.get_valid_callees(
            "orchestrator", self.active_specialist_ids
        )
        valid_ids = [sid for sid in valid_ids if sid not in self.called_ids]
        mask = self.action_decoder.build_specialist_mask(valid_ids)

        factored: FactoredAction = self.action_decoder.decode(action, mask)

        assert self._task_emb is not None, (
            "step() called before reset() or task embedding failed in reset()"
        )
        task_emb = self._task_emb

        terminated = False
        truncated = False
        step_results = []

        if factored.meta_action == MetaAction.STOP or self.step_count >= self.max_steps:
            terminated = True
        else:
            step_results = self._dispatch_meta_action(factored, elapsed_ms)
            self.specialist_results.extend(step_results)
            _reg = set(self.registry.list_ids())
            self.called_ids.extend(
                r.specialist_id for r in step_results
                if r.specialist_id in _reg
            )

        if self.step_count >= self.max_steps and not terminated:
            truncated = True
        state = build_state(
            task_embedding=task_emb,
            registry=self.registry,
            called_ids=self.called_ids,
            delegation_graph=self.delegation_graph,
            scratchpad=self.scratchpad,
            step_count=self.step_count,
            elapsed_ms=elapsed_ms,
            sla_budget_ms=self.latency_sla.budget_ms,
            max_specialists=self.max_specialists,
            max_depth=self.max_depth,
            phase=self.phase,
            active_ids=self.active_specialist_ids,
        )

        if terminated or truncated:
            reward = self._compute_final_reward(elapsed_ms)
            self._record_episode_to_memory(reward)
        else:
            reward = self._compute_step_reward(
                step_results, task_emb,
                delegation_mode=factored.delegation_mode,
                meta_action=factored.meta_action,
            )

        step_latencies = {r.specialist_id: r.latency_ms for r in step_results}
        info = {
            # Keys expected by the UI / Streamlit dashboard
            "action_name":           factored.meta_action.name,
            "called_specialists":    list(factored.specialist_ids),
            "delegation_mode":       factored.delegation_mode.name,
            "reward_components":     dict(self._last_reward_components),
            "specialist_latencies":  step_latencies,
            "active_specialists":    list(self.active_specialist_ids),
            "spawned_specialists":   list(self.spawned_this_episode),
            # Raw data for debugging / training callbacks
            "action":                factored.to_log_dict(),
            "called_ids":            list(self.called_ids),
            "step_count":            self.step_count,
            "elapsed_ms":            elapsed_ms,
        }

        return state.to_flat_vector(), reward, terminated, truncated, info

    # ── MetaAction dispatch ───────────────────────────────────────────

    def _dispatch_meta_action(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Route to the correct handler based on MetaAction."""
        if action.meta_action == MetaAction.CALL_MEDIATOR:
            return self._exec_meta_mediator(action, elapsed_ms)
        if action.meta_action == MetaAction.CLARIFY_TASK:
            return self._exec_meta_clarify(action, elapsed_ms)
        if action.meta_action == MetaAction.DELEGATE_SUBTASK:
            return self._exec_meta_delegate_subtask(action, elapsed_ms)
        if action.meta_action == MetaAction.RETRY_FAILED:
            return self._exec_meta_retry(action, elapsed_ms)
        if action.meta_action == MetaAction.PARALLEL_SPAWN:
            return self._exec_meta_parallel_spawn(action, elapsed_ms)
        if action.meta_action == MetaAction.SPAWN_SPECIALIST:
            return self._exec_meta_spawn_specialist(action, elapsed_ms)
        return self._execute_action(action, elapsed_ms)   # CALL_SPECIALIST default

    # ── DelegationMode dispatch ───────────────────────────────────────

    def _execute_action(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Dispatch to the correct DelegationMode handler."""
        handlers = {
            DelegationMode.SEQUENTIAL:     self._exec_sequential,
            DelegationMode.PARALLEL:       self._exec_parallel,
            DelegationMode.FAN_OUT_REDUCE: self._exec_fan_out_reduce,
            DelegationMode.ITERATIVE:      self._exec_iterative,
            DelegationMode.CONDITIONAL:    self._exec_conditional,
            DelegationMode.PRIORITY_QUEUE: self._exec_priority_queue,
            DelegationMode.BROADCAST:      self._exec_broadcast,
        }
        return handlers.get(action.delegation_mode, self._exec_sequential)(action, elapsed_ms)

    # ── Shared helpers ────────────────────────────────────────────────

    def _can_call(self, sid: str, caller_id: str = "orchestrator") -> bool:
        """True when a specialist is registered, not yet called, and DAG-valid."""
        return (
            sid in self.registry.list_ids()
            and sid not in self.called_ids
            and self.delegation_graph.can_delegate(caller_id, sid)
        )

    def _do_call(
        self,
        sid: str,
        task: str,
        elapsed_ms: float,
        mode: str = "SEQUENTIAL",
        context: str | None = None,
        caller_id: str = "orchestrator",
    ) -> list[SpecialistResult]:
        """
        Validate → record in DAG → call specialist → handle fallback → write scratchpad.

        caller_id controls which node in the delegation graph is the caller.
        Defaults to "orchestrator" for top-level calls. Pass a specialist ID
        to record depth-2 delegations (specialist → sub-specialist).
        Returns a list because a fallback may contribute a second result.
        """
        if not self._can_call(sid, caller_id=caller_id):
            return []
        self.delegation_graph.record_delegation(caller_id, sid, mode)
        result = self._call_specialist(sid, task, elapsed_ms, context=context)
        if result.output:
            self.scratchpad.write(
                author_id=sid,
                author_role=self.registry.get(sid).role,
                content=result.output,
            )
        results = [result]
        if self.fallback_resolver.needs_fallback(result):
            fb_id = self.fallback_resolver.get_fallback(sid, self.called_ids)
            if fb_id and self._can_call(fb_id):
                self.delegation_graph.record_delegation("orchestrator", fb_id, mode)
                fb = self._call_specialist(
                    fb_id, self.current_task.enriched_description, elapsed_ms
                )
                fb.fallback_used = True
                if fb.output:
                    self.scratchpad.write(
                        author_id=fb_id,
                        author_role=self.registry.get(fb_id).role,
                        content=fb.output,
                    )
                results.append(fb)
                # Do NOT append fb_id here — step() uniformly extends called_ids
                # from all step_results after _do_call returns, so appending here
                # would cause a double-count (efficiency penalty and DAG mask both
                # use called_ids, making the fallback specialist appear called twice).
        return results

    def _quick_quality_score(self, output: str, task: str) -> float:
        """Fast T1 cosine similarity — used for within-step stopping conditions."""
        try:
            t = self.registry.embed_query(task)
            o = self.registry.embed_query(output[:800])
            return float((self.registry.cosine_similarity(t, o) + 1.0) / 2.0)
        except Exception:
            return 0.5

    def _synthesize_outputs(self, outputs: list[str]) -> str:
        """Merge multiple specialist outputs into one coherent synthesis."""
        import os
        if os.getenv("OPENAI_API_KEY") and len(outputs) >= 2:
            try:
                from openai import OpenAI
                combined = "\n\n---\n\n".join(
                    f"Specialist {i+1}:\n{o[:500]}" for i, o in enumerate(outputs)
                )
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=600,
                    messages=[
                        {"role": "system", "content":
                            "Synthesize these specialist analyses into one coherent "
                            "recommendation. Resolve contradictions, highlight consensus."},
                        {"role": "user", "content": combined[:2000]},
                    ],
                )
                return resp.choices[0].message.content
            except Exception as exc:
                print(f"[Synthesize] {exc}")
        joined = "\n\n".join(f"[{i+1}] {o[:200]}" for i, o in enumerate(outputs))
        return (
            f"Synthesis of {len(outputs)} specialist outputs:\n{joined}\n"
            "Consensus: structured design, domain best practices, iterative validation."
        )

    # ── DelegationMode handlers ───────────────────────────────────────

    def _exec_sequential(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """A→B→C: each specialist receives accumulated context from prior outputs.
        Highest quality for dependent sub-problems."""
        results: list[SpecialistResult] = []
        context = ""
        for sid in action.specialist_ids:
            batch = self._do_call(
                sid, self.current_task.enriched_description,
                elapsed_ms, mode="SEQUENTIAL",
                context=context or None,
            )
            results.extend(batch)
            for r in batch:
                if r.output:
                    context += f"\n{r.output[:400]}"
        return results

    def _exec_parallel(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """All specialists see the same task independently — no context sharing.
        Lower quality than SEQUENTIAL, lower effective latency for independent work."""
        results: list[SpecialistResult] = []
        for sid in action.specialist_ids:
            results.extend(
                self._do_call(
                    sid, self.current_task.enriched_description,
                    elapsed_ms, mode="PARALLEL",
                )
            )
        return results

    def _exec_fan_out_reduce(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Fan-out: all specialists run independently; reduce: a synthesis pass
        merges all outputs into one recommendation. Highest quality, highest cost."""
        results = self._exec_parallel(action, elapsed_ms)
        successful_outs = [
            r.output for r in results
            if r.status == SpecialistStatus.SUCCESS and r.output
        ]
        if len(successful_outs) >= 2:
            synthesis = self._synthesize_outputs(successful_outs)
            synth = SpecialistResult(
                specialist_id="synthesizer",
                status=SpecialistStatus.SUCCESS,
                output=synthesis,
                latency_ms=0.0,
            )
            self.scratchpad.write(
                author_id="synthesizer",
                author_role="Synthesis Mediator",
                content=synthesis,
            )
            results.append(synth)
        return results

    def _exec_iterative(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Repeatedly call one specialist, feeding its output back as context,
        until quality threshold met or max_rounds exhausted."""
        if not action.specialist_ids:
            return []
        sid        = action.specialist_ids[0]
        max_rounds = int(action.mode_params.get("max_rounds", 3))
        threshold  = float(action.mode_params.get("quality_threshold", 0.70))
        results: list[SpecialistResult] = []
        context = ""
        for _ in range(max(1, max_rounds)):
            batch = self._do_call(
                sid, self.current_task.enriched_description,
                elapsed_ms, mode="ITERATIVE",
                context=context or None,
            )
            results.extend(batch)
            for r in batch:
                if r.output:
                    if self._quick_quality_score(r.output, self.current_task.enriched_description) >= threshold:
                        return results
                    context = r.output
        return results

    def _exec_conditional(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Call specialists in order; stop as soon as one meets the quality
        threshold — avoids unnecessary calls when the first is sufficient."""
        threshold = float(action.mode_params.get("condition_threshold", 0.60))
        results: list[SpecialistResult] = []
        for sid in action.specialist_ids:
            batch = self._do_call(
                sid, self.current_task.enriched_description,
                elapsed_ms, mode="CONDITIONAL",
            )
            results.extend(batch)
            for r in batch:
                if r.output and self._quick_quality_score(
                    r.output, self.current_task.enriched_description
                ) >= threshold:
                    return results
        return results

    def _exec_priority_queue(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Sort selected specialists by task-similarity, call highest-ranked first,
        stop when output quality meets stop_threshold. Good for SLA-sensitive tasks."""
        threshold = float(action.mode_params.get("stop_threshold", 0.70))
        task_emb  = self.registry.embed_query(self.current_task.enriched_description)
        sorted_sids = sorted(
            [sid for sid in action.specialist_ids if self._can_call(sid)],
            key=lambda s: (
                self.registry.cosine_similarity(
                    task_emb, self.registry.get(s).to_state_vector()
                ) if s in self.registry.list_ids() else 0.0
            ),
            reverse=True,
        )
        results: list[SpecialistResult] = []
        for sid in sorted_sids:
            batch = self._do_call(
                sid, self.current_task.enriched_description,
                elapsed_ms, mode="PRIORITY_QUEUE",
            )
            results.extend(batch)
            for r in batch:
                if r.output and self._quick_quality_score(
                    r.output, self.current_task.enriched_description
                ) >= threshold:
                    return results
        return results

    def _exec_broadcast(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Call all specialists independently, return only the single best result.
        Trades extra API calls for a quality ceiling guarantee."""
        results = self._exec_parallel(action, elapsed_ms)
        successful = [
            r for r in results
            if r.status == SpecialistStatus.SUCCESS and r.output
        ]
        if not successful:
            return results
        best = max(
            successful,
            key=lambda r: self._quick_quality_score(
                r.output, self.current_task.enriched_description
            ),
        )
        self.scratchpad.write(
            author_id=best.specialist_id,
            author_role=(
                self.registry.get(best.specialist_id).role
                if best.specialist_id in self.registry.list_ids() else "Specialist"
            ),
            content=f"[BROADCAST WINNER]\n{best.output}",
        )
        return [best]

    # ── MetaAction handlers ───────────────────────────────────────────

    def _exec_meta_mediator(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Synthesise all current specialist_results to resolve conflicts.
        Only meaningful after ≥2 specialist outputs exist this episode."""
        outputs = [
            r.output for r in self.specialist_results
            if r.status == SpecialistStatus.SUCCESS and r.output
        ]
        if len(outputs) < 2:
            return []
        synthesis = self._synthesize_outputs(outputs)
        result = SpecialistResult(
            specialist_id="mediator",
            status=SpecialistStatus.SUCCESS,
            output=synthesis,
            latency_ms=0.0,
        )
        self.scratchpad.write(
            author_id="mediator", author_role="Conflict Mediator", content=synthesis
        )
        return [result]

    def _exec_meta_clarify(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Enrich the current task description (via LLM when key available).
        All future specialist calls in this episode see the richer description."""
        import os
        original = self.current_task.enriched_description
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=250,
                    messages=[
                        {"role": "system", "content":
                            "Expand this task into a more specific, actionable description. "
                            "Add missing technical context. Keep it under 3 sentences."},
                        {"role": "user", "content": original[:500]},
                    ],
                )
                clarified = resp.choices[0].message.content.strip()
            except Exception as exc:
                print(f"[ClarifyTask] {exc}")
                clarified = original + " [Clarified: requires structured design and domain-specific approach]"
        else:
            clarified = (
                original + " [Clarified: requires structured design, "
                "clear acceptance criteria, and a domain-specific technical approach]"
            )
        self.current_task = type(self.current_task)(
            original_description=self.current_task.original_description,
            enriched_description=clarified,
            complexity_class=self.current_task.complexity_class,
            expected_specialists=self.current_task.expected_specialists,
            domain_hints=self.current_task.domain_hints,
            is_ambiguous=False,
            autonomously_enriched=True,
        )
        self.scratchpad.write(
            author_id="orchestrator", author_role="Orchestrator",
            content=f"Task clarified: {clarified[:300]}",
        )
        self._task_emb = self.registry.embed_query(clarified)
        return []  # effect is through improved quality on future specialist calls

    def _exec_meta_delegate_subtask(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Decompose the task into 2–3 subtasks and route each to the best-matching
        sub-specialist, with the lead specialist as the DAG caller (depth 1→2).

        This is the only execution path that produces depth > 1 in the delegation
        graph.  The first specialist in action.specialist_ids acts as the delegating
        node; its sub-calls are recorded as specialist → sub-specialist edges so
        self.delegation_graph.depth reaches 2 when max_depth=2 permits it.
        """
        import os, json
        task = self.current_task.enriched_description

        # ── Step 1: call the lead specialist at depth 1 (orchestrator → lead) ──
        lead_id = next(
            (sid for sid in action.specialist_ids if self._can_call(sid, "orchestrator")),
            None,
        )
        results: list[SpecialistResult] = []
        if lead_id:
            results.extend(self._do_call(lead_id, task, elapsed_ms,
                                         mode="DELEGATE_SUBTASK", caller_id="orchestrator"))
        # If no lead could be called, fall through to sequential
        if not lead_id:
            return self._exec_sequential(action, elapsed_ms)

        # ── Step 2: decompose into subtasks ──
        subtasks: list[str] = []
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=250,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content":
                            "Break this task into 2-3 distinct subtasks. "
                            "Return JSON: {\"subtasks\": [\"subtask1\", ...]}"},
                        {"role": "user", "content": task[:500]},
                    ],
                )
                subtasks = json.loads(resp.choices[0].message.content).get("subtasks", [])[:3]
            except Exception as exc:
                print(f"[DelegateSubtask] {exc}")
        if not subtasks:
            subtasks = [
                f"{task[:200]} — part 1: design and requirements",
                f"{task[:200]} — part 2: implementation and validation",
            ]

        # ── Step 3: route each subtask from lead_id (depth 1 → 2) ──
        for subtask in subtasks:
            sub_emb = self.registry.embed_query(subtask)
            for sid, _ in self.registry.find_most_similar(sub_emb, top_k=self.max_specialists):
                if self._can_call(sid, caller_id=lead_id):
                    results.extend(self._do_call(sid, subtask, elapsed_ms,
                                                 mode="DELEGATE_SUBTASK", caller_id=lead_id))
                    break
        return results

    def _exec_meta_retry(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Retry all failed/timed-out specialist calls using the FallbackChainResolver."""
        failed = [r for r in self.specialist_results if r.status != SpecialistStatus.SUCCESS]
        if not failed:
            return []
        results: list[SpecialistResult] = []
        for fr in failed:
            fb_id = self.fallback_resolver.get_fallback(fr.specialist_id, self.called_ids)
            if fb_id and self._can_call(fb_id):
                batch = self._do_call(
                    fb_id, self.current_task.enriched_description,
                    elapsed_ms, mode="RETRY_FAILED",
                )
                for r in batch:
                    r.fallback_used = True
                results.extend(batch)
        return results

    def _exec_meta_parallel_spawn(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """Spawn all selected specialists in parallel (delegates to PARALLEL mode)."""
        return self._exec_parallel(action, elapsed_ms)

    # ── Roster management ─────────────────────────────────────────────

    def _select_active_specialists(self, task_emb: np.ndarray) -> list[str]:
        """
        Pick the max_specialists agents most relevant to this task.
        Always ensures any specialist spawned this episode is in the set.
        """
        ranked = self.registry.find_most_similar(
            task_emb, top_k=self.registry.size
        )
        selected = [sid for sid, _ in ranked[: self.max_specialists]]

        # Guarantee newly spawned specialists are in the active window
        for sid in self.spawned_this_episode:
            if sid not in selected:
                selected[-1] = sid  # replace least-relevant

        return selected

    def _exec_meta_spawn_specialist(
        self, action: FactoredAction, elapsed_ms: float
    ) -> list[SpecialistResult]:
        """
        Policy-triggered specialist spawn.
        Guards: OPENAI_API_KEY required, cooldown and total cap enforced.
        After a successful spawn the active roster and action decoder are
        refreshed so the new specialist is immediately selectable.
        """
        import os
        task_desc = self.current_task.enriched_description

        # Guard: no API key
        if not os.getenv("OPENAI_API_KEY"):
            return []

        # Guard: total cap
        if self._spawn_total_count >= self._spawn_max_total:
            return []

        # Guard: cooldown
        episodes_since_last = self._episode_index - self._last_spawn_episode
        if episodes_since_last < self._spawn_cooldown_episodes:
            return []

        # All guards passed — attempt spawn
        prev_count = self._spawn_total_count
        top1 = self.registry.find_most_similar(self._task_emb, top_k=1)
        best_id  = top1[0][0] if top1 else ""
        best_sim = top1[0][1] if top1 else 0.0
        self._spawn_via_llm(task_desc, best_sim=best_sim, best_id=best_id)

        if self._spawn_total_count > prev_count:
            new_id = self.spawned_this_episode[-1]
            # Refresh active roster so the new specialist is immediately reachable
            self.active_specialist_ids = self._select_active_specialists(self._task_emb)
            self.action_decoder = ActionDecoder(
                specialist_ids=self.active_specialist_ids,
                max_specialists=self.max_specialists,
            )
            return [SpecialistResult(
                specialist_id=new_id,
                status=SpecialistStatus.SUCCESS,
                output=f"[SpawnSpecialist] Spawned '{new_id}' successfully.",
                latency_ms=0.0,
            )]
        else:
            return [SpecialistResult(
                specialist_id="spawn_attempt",
                status=SpecialistStatus.ERROR,
                output="[SpawnSpecialist] LLM spawn failed — see logs.",
                latency_ms=0.0,
            )]

    def _maybe_spawn_specialist(
        self, task_emb: np.ndarray, task: str
    ) -> None:
        """
        Auto-spawn a new specialist via LLM when the best existing match
        falls below spawn_threshold.  Skipped when no OPENAI_API_KEY.
        """
        top1 = self.registry.find_most_similar(task_emb, top_k=1)
        if not top1:
            return
        best_id, best_sim = top1[0]
        if best_sim >= self.spawn_threshold:
            return  # roster already covers the task well enough
        self._spawn_via_llm(task, best_sim, best_id)

    def _spawn_via_llm(
        self, task: str, best_sim: float, best_id: str
    ) -> None:
        """
        Ask GPT-4o-mini to design a new specialist for this task,
        then add it to the registry so it enters the active roster.
        Conditions the prompt on past successful spawns for similar tasks.
        """
        import os, json
        existing_roles = [self.registry.get(s).role for s in self.registry.list_ids()]
        best_role = self.registry.get(best_id).role if best_id else "none"

        # Retrieve similar past successful spawns for RAG context
        min_reward = self.config.get("environment", {}).get("spawn_memory_min_reward", 0.0)
        past_spawns = self._spawn_memory.retrieve_similar(
            self._task_emb, top_k=3, min_reward=min_reward
        )
        past_context = ""
        if past_spawns:
            examples = "\n".join(
                f"- Role: {r.specialist_role}  |  "
                f"Desc: {r.specialist_desc[:150]}  |  "
                f"Reward: {r.episode_reward:.2f}"
                for r in past_spawns
            )
            past_context = (
                f"\n\nPast successful spawns for similar tasks:\n{examples}\n"
                "Use these as inspiration but create something distinct if needed."
            )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=350,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You design specialist agent definitions for a multi-agent "
                            "delegation system. Return valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Task: {task[:400]}\n\n"
                            f"Existing specialists: {', '.join(existing_roles)}\n"
                            f"Best current match: {best_role} "
                            f"(cosine similarity {best_sim:.2f} — below threshold)."
                            f"{past_context}\n\n"
                            "Define a new specialist better suited to this task. "
                            "Return JSON with keys: id (snake_case), role (title case), "
                            "description (2–3 sentences of domain expertise), "
                            "complexity_affinity (list from [atomic,simple,moderate,complex,enterprise]), "
                            "avg_latency_ms (integer, 2000–8000)."
                        ),
                    },
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            required = {"id", "role", "description", "complexity_affinity", "avg_latency_ms"}
            if not required.issubset(data):
                print(f"[SpawnSpecialist] Incomplete JSON: {data}")
                return
            # Deduplicate ID
            base_id = str(data["id"]).lower().replace(" ", "_")
            uid = base_id
            suffix = 2
            while uid in self.registry.list_ids():
                uid = f"{base_id}_v{suffix}"
                suffix += 1
            data["id"] = uid
            self.registry.add_specialist(data)
            self.spawned_this_episode.append(uid)
            self._spawn_total_count += 1
            self._last_spawn_episode = self._episode_index
            print(
                f"[SpawnSpecialist] Created '{data['role']}' (id={uid}) "
                f"for task (best_sim was {best_sim:.2f}, "
                f"total spawned={self._spawn_total_count}/{self._spawn_max_total})"
            )
            # Stage a pending spawn record — reward filled in at episode end
            self._pending_spawn_records.append(SpawnRecord(
                task_embedding=self._task_emb.tolist(),
                task_description=task,
                specialist_id=uid,
                specialist_role=data["role"],
                specialist_desc=data["description"],
                episode_reward=0.0,    # filled in at episode end
                pre_spawn_sim=best_sim,
                post_spawn_sim=0.0,    # filled after re-ranking
                episode_idx=self._episode_index,
            ))
        except Exception as exc:
            print(f"[SpawnSpecialist] Failed: {exc}")

    # ── Specialist execution ───────────────────────────────────────────

    def _call_specialist(
        self, specialist_id: str, task: str, elapsed_ms: float,
        context: str | None = None,
    ) -> SpecialistResult:
        """
        Call a specialist.
        Priority order:
          1. use_real_spindleflow=True  → TypeScript SpindleFlow subprocess
          2. OPENAI_API_KEY set         → real OpenAI call per specialist
          3. neither                    → fast simulation (training / offline)

        context: optional accumulated output from prior specialists (SEQUENTIAL/ITERATIVE).
        """
        import os
        specialist = self.registry.get(specialist_id)

        if self.use_real_spindleflow:
            output, latency, status = self._call_real_spindleflow(specialist_id, task)
        elif os.getenv("OPENAI_API_KEY") and not self.simulate_specialists:
            output, latency, status = self._call_openai_specialist(specialist_id, task, context=context)
        else:
            output  = self._simulate_specialist_output(specialist_id, task, context=context)
            latency = specialist.avg_latency_ms + np.random.normal(0, 500)
            status  = SpecialistStatus.SUCCESS

        return SpecialistResult(
            specialist_id=specialist_id,
            status=status,
            output=output,
            latency_ms=max(0, latency),
        )

    def _call_openai_specialist(
        self, specialist_id: str, task: str,
        context: str | None = None,
    ) -> tuple[str, float, SpecialistStatus]:
        """Call GPT-4o-mini acting as this specialist. Each gets its own system prompt.

        context: prior specialist output (SEQUENTIAL/ITERATIVE). When present, injected
                 as a user/assistant exchange before the current task so the model builds
                 on accumulated analysis rather than starting fresh.
        """
        import os
        specialist = self.registry.get(specialist_id)
        start = time.time()
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if specialist.system_prompt:
                system_content = specialist.system_prompt
            else:
                system_content = (
                    f"You are a {specialist.role}. {specialist.description} "
                    f"Give a focused, expert response relevant to your specialty."
                )
            messages = [{"role": "system", "content": system_content}]
            if context:
                messages.append({
                    "role": "user",
                    "content": f"Prior specialist analysis:\n{context[:600]}",
                })
                messages.append({
                    "role": "assistant",
                    "content": "Understood. I'll build on this prior analysis.",
                })
            messages.append({"role": "user", "content": f"Task: {task[:600]}"})
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=600,
                messages=messages,
            )
            latency = (time.time() - start) * 1000
            return response.choices[0].message.content, latency, SpecialistStatus.SUCCESS
        except Exception as exc:
            latency = (time.time() - start) * 1000
            print(f"[OpenAI specialist {specialist_id}] Error: {exc}")
            return "", latency, SpecialistStatus.ERROR

    def _simulate_specialist_output(
        self, specialist_id: str, task: str,
        context: str | None = None,
    ) -> str:
        """
        Simulate specialist output for training (no API key).

        Critically: the task text is NOT embedded in the output.
        Output quality is driven entirely by domain vocabulary from the
        specialist description, which naturally correlates with the task
        embedding when the specialist is a good match. This gives T1
        quality_delta a real signal (specialist–task domain overlap)
        rather than the degenerate case where both sides quote task[:100]
        and collapse quality_delta to noise.

        context: prior specialist output (SEQUENTIAL/ITERATIVE). When present and
                 similarity is high, the output acknowledges and extends prior work.

        Three quality tiers based on specialist-task cosine similarity:
          > 0.45  → rich domain analysis (high T1 score if relevant)
          > 0.25  → partial domain guidance
          ≤ 0.25  → mismatched — minimal domain content (low T1 score)
        """
        specialist = self.registry.get(specialist_id)
        task_emb = self.registry.embed_query(task)
        spec_emb = specialist.to_state_vector()
        similarity = self.registry.cosine_similarity(task_emb, spec_emb)

        context_prefix = ""
        if context and similarity > 0.45:
            context_prefix = (
                f"Building on the prior analysis, I will extend with {specialist.role.lower()} "
                f"expertise.\n"
            )

        if similarity > 0.45:
            return (
                f"{context_prefix}As a {specialist.role}, here is my expert analysis.\n"
                f"{specialist.description}\n"
                f"Key technical considerations from this domain: systematic design, "
                f"stakeholder alignment, iterative validation, and rigorous testing. "
                f"I recommend applying established {specialist.role.lower()} frameworks "
                f"with particular attention to quality gates and domain-specific constraints."
            )
        elif similarity > 0.25:
            return (
                f"As a {specialist.role}, I can provide partial guidance. "
                f"My expertise: {specialist.description[:200]}. "
                f"For aspects outside my specialty, additional expert input is recommended."
            )
        else:
            return (
                f"As a {specialist.role}, this request falls largely outside my primary domain. "
                f"I can offer only general guidance and recommend a more suitable specialist."
            )

    def _call_real_spindleflow(
        self, specialist_id: str, task: str
    ) -> tuple[str, float, SpecialistStatus]:
        """
        Call the real SpindleFlow TypeScript backend via subprocess.
        Returns (output, latency_ms, status).
        """
        import subprocess
        import json
        import os
        import tempfile

        spindleflow_path = os.getenv("SPINDLEFLOW_PATH", "../SpindleFlow")
        specialist = self.registry.get(specialist_id)

        config = {
            "models": {
                "gemini": {
                    "provider": "gemini",
                    "model": "gemini-2.5-flash-lite",
                    "max_tokens": 4096,
                }
            },
            "provider": "gemini",
            "agents": [{
                "id": specialist_id,
                "role": specialist.role,
                "goal": specialist.description,
            }],
            "workflow": {
                "type": "sequential",
                "steps": [{"agent": specialist_id}],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            yaml.dump(config, f)
            config_path = f.name

        start = time.time()
        try:
            result = subprocess.run(
                ["npm", "run", "dev", "--", "run", config_path, "-i", task[:500]],
                cwd=spindleflow_path,
                capture_output=True,
                text=True,
                timeout=self.specialist_timeout_ms / 1000,
            )
            latency = (time.time() - start) * 1000
            if result.returncode == 0:
                output = result.stdout[-2000:]
                return output, latency, SpecialistStatus.SUCCESS
            else:
                return "", latency, SpecialistStatus.ERROR
        except subprocess.TimeoutExpired:
            latency = (time.time() - start) * 1000
            return "", latency, SpecialistStatus.TIMEOUT
        finally:
            try:
                os.unlink(config_path)
            except Exception:
                pass

    def _generate_generalist_baseline(self, task: str) -> str:
        """
        Generate a generalist (non-specialist) response to the task.
        Uses OpenAI when OPENAI_API_KEY is set (regardless of use_real_spindleflow).
        Falls back to a simulated template when no key is available.
        """
        import os
        if getattr(self, "simulate_specialists", False) or not os.getenv("OPENAI_API_KEY"):
            return (
                "General problem-solving approach:\n"
                "1. Gather and clarify requirements\n"
                "2. Research common solution patterns\n"
                "3. Draft a high-level architecture\n"
                "4. Implement in small, testable increments\n"
                "5. Validate against acceptance criteria and deploy\n"
                "No specialist domain expertise applied."
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=500,
                    messages=[{"role": "user", "content": f"Please help with: {task}"}],
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[Baseline] OpenAI error: {e}. Using simulated baseline.")
        # Simulation baseline: domain-neutral boilerplate, NO task text.
        # Must embed far from any specific task so quality_delta is positive
        # whenever a matched specialist contributes domain-relevant content.
        return (
            "General problem-solving approach:\n"
            "1. Gather and clarify requirements\n"
            "2. Research common solution patterns\n"
            "3. Draft a high-level architecture\n"
            "4. Implement in small, testable increments\n"
            "5. Validate against acceptance criteria and deploy\n"
            "No specialist domain expertise applied."
        )

    def _compute_step_reward(
        self,
        step_results: list[SpecialistResult],
        task_emb: np.ndarray,
        delegation_mode: "DelegationMode | None" = None,
        meta_action: "MetaAction | None" = None,
    ) -> float:
        """
        Per-step shaping reward for non-terminal steps.

        Base shaping:
          +0.02  per specialist whose cosine-sim with task > 0.35  (good routing)
          -0.01  per specialist below 0.20                          (mismatch)
          -0.01  per failed call

        Mode-specific adjustments (make mode choice matter before terminal reward):

          PARALLEL — specialists ran concurrently; effective wall-clock cost is
            max(latencies) not sum(latencies).  Reward the latency saving when
            ≥2 specialists ran: +0.01 * (1 - max_lat / sum_lat).
            E.g. 3 specialists × 1 s each → sum=3 s, max=1 s → saving=0.67 →
            bonus ≈ +0.0067.  Scales to zero when only one specialist runs.

          SEQUENTIAL — scratchpad-chaining means each specialist built on prior
            output.  Reward the coordination effort: +0.01 per specialist after
            the first one (they had real context to work with), capped at +0.03.

        Scale stays small vs terminal range [-1, 2] so episode quality_delta
        dominates.  Total step shaping over 10 steps tops out at ~0.25.
        """
        if not step_results or not self.current_task:
            self._last_reward_components = {"step_shaping": 0.0}
            return 0.0

        shaped = 0.0
        for result in step_results:
            if result.status != SpecialistStatus.SUCCESS:
                shaped -= 0.01
                continue
            if result.specialist_id not in self.registry.list_ids():
                continue
            spec_emb = self.registry.get(result.specialist_id).to_state_vector()
            sim = self.registry.cosine_similarity(task_emb, spec_emb)
            if sim > 0.35:
                shaped += 0.02
            elif sim < 0.20:
                shaped -= 0.01

        # Mode-specific bonus
        mode_bonus = 0.0
        successful = [r for r in step_results if r.status == SpecialistStatus.SUCCESS]
        if delegation_mode == DelegationMode.PARALLEL and len(successful) >= 2:
            latencies = [r.latency_ms for r in successful]
            sum_lat = sum(latencies)
            if sum_lat > 0:
                saving = 1.0 - max(latencies) / sum_lat
                mode_bonus = round(0.01 * saving, 4)
        elif delegation_mode == DelegationMode.SEQUENTIAL and len(successful) >= 2:
            # Each specialist after the first had chained context
            chained_count = len(successful) - 1
            mode_bonus = min(0.01 * chained_count, 0.03)

        shaped += mode_bonus

        # Spawn quality shaping — only on SPAWN_SPECIALIST steps
        spawn_bonus = 0.0
        if meta_action == MetaAction.SPAWN_SPECIALIST:
            spawn_succeeded = any(
                r.status == SpecialistStatus.SUCCESS
                and r.specialist_id in self.spawned_this_episode
                for r in step_results
            )
            if spawn_succeeded:
                new_id = self.spawned_this_episode[-1]
                try:
                    new_spec_vec = self.registry.get(new_id).to_state_vector()
                    new_sim = float(self.registry.cosine_similarity(task_emb, new_spec_vec))
                    # Reward coverage gap closed above threshold; penalise redundant spawns
                    spawn_bonus = round(0.05 * max(0.0, new_sim - self.spawn_threshold), 4)
                except Exception:
                    spawn_bonus = 0.0
            else:
                # Guard hit or LLM failed — mild penalty to discourage wasteful spawn attempts
                spawn_bonus = -0.02

        shaped += spawn_bonus
        self._last_reward_components = {
            "step_shaping": float(shaped),
            "mode_bonus": float(mode_bonus),
            "spawn_bonus": float(spawn_bonus),
        }
        return float(shaped)

    def _compute_final_reward(self, elapsed_ms: float) -> float:
        """Compute the full reward for a completed episode."""
        _zero = {k: 0.0 for k in [
            "quality_delta", "efficiency_penalty", "failure_penalty",
            "recovery_bonus", "conflict_penalty", "conflict_bonus",
            "consistency_bonus", "latency_penalty", "explanation_bonus",
        ]}
        if not self.specialist_results or not self.current_task:
            self._last_reward_components = {**_zero, "failure_penalty": -0.1}
            return -0.1

        successful_outputs = [
            r.output for r in self.specialist_results
            if r.status == SpecialistStatus.SUCCESS and r.output
        ]

        if not successful_outputs:
            self._last_reward_components = {**_zero, "failure_penalty": -0.2}
            return -0.2

        specialist_output = "\n\n".join(successful_outputs)
        task_desc = self.current_task.enriched_description

        # Delta reward — same tier for both
        specialist_score = self.reward_scorer.score(
            specialist_output, task_desc, self.tier_lock
        )
        baseline_score = self.reward_scorer.score(
            self.generalist_baseline, task_desc, self.tier_lock
        )
        quality_delta = specialist_score - baseline_score

        # Efficiency penalty
        n = len(self.called_ids)
        expected = self.current_task.expected_specialists
        efficiency_penalty = self.config_reward["efficiency_base_penalty"] * \
                             max(0, n - expected)

        # Failure signals
        failure_penalty = compute_failure_penalty(self.specialist_results)
        recovery_bonus = compute_recovery_bonus(
            self.specialist_results, episode_completed=True
        )

        # Conflict signals
        conflicts = detect_conflicts(
            self.specialist_results,
            registry=self.registry,
            contradiction_pairs=self._contradiction_pairs,
            similarity_threshold=self.config_reward.get(
                "conflict_similarity_threshold", 0.25
            ),
        )
        if conflicts:
            self.conflict_resolver.resolve_all(conflicts, self.specialist_results)
        conflict_penalty = self.config_reward["conflict_unresolved_penalty"] * \
                          len([c for c in conflicts if not c.resolved])
        conflict_bonus = self.config_reward["conflict_resolved_bonus"] * \
                        len([c for c in conflicts if c.resolved])

        # Consistency bonus
        path = self.delegation_graph.get_delegation_path()
        consistency = self.consistency_tracker.consistency_score(
            path, self.current_task.complexity_class
        )
        consistency_bonus = self.config_reward["consistency_bonus_weight"] * consistency

        # Latency penalty
        latency_penalty = compute_latency_penalty(elapsed_ms, self.latency_sla)

        # Explanation bonus
        explanation_bonus = (
            self.config_reward["explanation_bonus"]
            if self.delegation_graph.is_auditable()
            else 0.0
        )

        self.consistency_tracker.record_path(
            self.current_task.complexity_class, path
        )

        total_reward = (
            quality_delta
            - efficiency_penalty
            - failure_penalty
            + recovery_bonus
            - conflict_penalty
            + conflict_bonus
            + consistency_bonus
            - latency_penalty
            + explanation_bonus
        )

        self._last_reward_components = {
            "quality_delta":      float(quality_delta),
            "efficiency_penalty": float(-efficiency_penalty),
            "failure_penalty":    float(-failure_penalty),
            "recovery_bonus":     float(recovery_bonus),
            "conflict_penalty":   float(-conflict_penalty),
            "conflict_bonus":     float(conflict_bonus),
            "consistency_bonus":  float(consistency_bonus),
            "latency_penalty":    float(-latency_penalty),
            "explanation_bonus":  float(explanation_bonus),
        }

        total_reward_clipped = float(np.clip(total_reward, -1.0, 2.0))

        # Record conflict resolution outcomes so the bandit can learn
        self.conflict_resolver.record_episode_outcome(
            quality_delta=float(quality_delta),
            episode_idx=self._episode_index,
        )

        # Finalise pending spawn records with the actual episode reward
        if self._pending_spawn_records and self._task_emb is not None:
            top_post = self.registry.find_most_similar(self._task_emb, top_k=1)
            post_sim = top_post[0][1] if top_post else 0.0
            for rec in self._pending_spawn_records:
                rec.episode_reward = total_reward_clipped
                rec.post_spawn_sim = post_sim
                self._spawn_memory.record(rec)
            self._pending_spawn_records = []

        return total_reward_clipped

    def _record_episode_to_memory(self, episode_reward: float) -> None:
        """Record each specialist's output and the episode reward to SpecialistMemory."""
        if not self.current_task:
            return
        task_desc = self.current_task.enriched_description
        for result in self.specialist_results:
            if result.specialist_id in self.spawned_this_episode:
                continue   # skip spawn confirmation messages
            if result.status == SpecialistStatus.SUCCESS and result.output:
                self.specialist_memory.record(
                    specialist_id=result.specialist_id,
                    task=task_desc,
                    output=result.output,
                    reward=episode_reward,
                )

    def render(self) -> None:
        if self.render_mode == "human" and self.current_task:
            print(f"\n[Episode State]")
            print(f"  Task: {self.current_task.enriched_description[:80]}")
            print(f"  Step: {self.step_count}/{self.max_steps}")
            print(f"  Called: {self.called_ids}")
            print(f"  Depth: {self.delegation_graph.depth}")

    def close(self) -> None:
        pass
