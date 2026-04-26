# SpindleFlow RL 

## The Problem That Started It All

Most multi-agent AI systems today have a gap that nobody talks about. You have an orchestrator and a roster of specialist agents. The orchestrator needs to decide who to call, when to call them, how to call them, and when to stop. Today, that logic is hand-written. Someone literally codes: "if the task mentions React, call the frontend agent." It breaks the moment you add a new agent, encounter an ambiguous task, or need to trade off quality against speed.

We thought the orchestrator should *learn* to conduct instead of being told how.

---

## Where The Story Begins

Our journey started at AuraVerse 2.0, a hackathon organized by the academic clubs at Scaler School of Technology. The challenge was deceptively simple: build a multi-agent system whose orchestration logic doesn't break when the agents change. No rewriting the routing rules. No hardcoded "if frontend, call frontend_react."

That question stuck with us. We carried it into today's challenge, extended it quite a bit, and built something we think is genuinely new: **SpindleFlow RL**, an environment where the orchestrator doesn't just execute delegation, it *learns* delegation as a skill.

---

## What SpindleFlow RL Actually Does

The core insight is this: delegation strategy is a sequential decision problem under partial observability. The right framework for that is reinforcement learning, specifically an LSTM policy (so the agent remembers what it has already called this episode) trained with PPO.

At every step of an episode, the orchestrator observes the task, the available specialist roster, what has already been called, and the shared scratchpad of accumulated analysis. It then decides across four dimensions simultaneously:

**What to do** — Call a specialist? Stop? Spawn a new one? Retry a failure? Mediate a conflict?

**Who to call** — Chosen not by ID, but by 384-dimensional capability embeddings. This is the key architectural decision: the policy never sees specialist names, only their semantic vectors. Add a new specialist and the policy represents it immediately, zero-shot, with no retraining.

**How to call them** — multiple delegation modes including sequential and parallel.

**When to stop** — A learned STOP action, not a hardcoded episode length. The policy decides when enough analysis has been done.

---

## The Five Things That Learn

This is where SpindleFlow RL goes beyond a simple routing experiment.

**1. Orchestration logic.** The LSTM PPO policy learns which specialists to call for which task types, in what order, and when to stop. The reward curve tells that story directly, covered in the numbers section below.

**2. Specialist quality.** Each specialist maintains a memory of its own past outputs and the episode rewards they produced. A SpecialistFinetuner periodically reads the high-reward examples (what worked) and the low-reward examples (what failed), then uses GPT-4o-mini to rewrite that specialist's system prompt. Every future call to that specialist uses the improved prompt. The specialists themselves get better as training progresses.

**3. Auto-spawning.** When the best existing specialist scores a cosine similarity below 0.40 against the task embedding (meaning the roster genuinely doesn't cover this domain) the environment asks GPT-4o-mini to design a new specialist from scratch. It generates an ID, role, description, and complexity affinity. The new specialist is embedded and added to the registry immediately, and the active roster for that episode is rebuilt to include it. A SpawnMemory records which spawns produced good episode rewards, and future spawns for similar tasks are conditioned on those past successes.

**4. Conflict resolution.** When two specialist outputs are semantically divergent (detected via cosine similarity below a configurable threshold, or via sector-defined contradiction pairs loaded from the catalog such as "postgresql vs mongodb") a ConflictResolver mediates. It selects resolution templates using an epsilon-greedy bandit that tracks which strategy produces the best quality delta per conflict type. The bandit's arm statistics persist across episodes in a JSONL file and improve over the course of training.

**5. Delegation mode selection.** The policy doesn't just learn *who* to call, it learns *how*. A quality-optimized policy routes five specialists sequentially. A latency-optimized policy, trained with a higher latency weight in the reward, routes three specialists in parallel. Same training infrastructure, different reward signal, measurably different behavior.

---

## The Reward Signal

Getting the reward right was honestly the hardest part of the design. The naive approach (score the output, subtract the baseline) has a fatal flaw: if you score the specialist output at Tier 2 with an LLM judge but the baseline at Tier 1 with embedding similarity, the delta is meaningless.

We solved this with an episode-level tier lock. At the start of each episode, based on task complexity, a tier is locked for both the specialist output *and* the generalist baseline. They are always scored through the same lens.

The full reward signal combines:

- **Quality delta** — specialist score minus generalist baseline, always the same tier
- **Efficiency penalty** — for calling more specialists than the task complexity warrants
- **Failure penalties** — for timeouts and errors, reduced if a fallback recovered successfully
- **Conflict penalties and bonuses** — unresolved conflicts penalized, resolved ones rewarded
- **Consistency bonus** — using a Dirichlet prior so the signal is non-zero from episode one
- **Latency penalty** — a tunable weight for time-sensitive deployments
- **Explanation bonus** — if the delegation graph is fully auditable

---

## What You Need to Change to Switch Domains

Two YAML files.

`specialist_catalog.yaml` defines your agents, their roles, descriptions, complexity affinities, and average latencies. The policy never sees the IDs; it operates on embeddings computed from the descriptions.

`training_config.yaml` defines your sector, the default technology assumptions, complexity keywords, and reward weights.

That's it. A healthcare deployment would swap in HL7/FHIR specialists. A legal platform would swap in contract analysis and compliance specialists. The RL policy, the reward architecture, the conflict resolution system, the spawn mechanism -- none of it changes.

---

## The Numbers

The learning curve tells the story directly. This is real training output, not a simulation.

We trained for **30,000 environment steps across approximately 13,400 episodes**, completing all three curriculum phases. The curriculum is performance-gated, not time-gated: the policy only advances from Phase 1 to Phase 2, and from Phase 2 to Phase 3, once its rolling mean reward crosses the threshold for that phase. This means Phase 1 and Phase 2 (atomic, simple, and moderate task delegation) were genuinely mastered before the policy was ever exposed to complex and enterprise-grade tasks. The agent earned its way to Phase 3.

The reward trajectory reflects this progression clearly. In the first ~500 episodes the smoothed reward drops to around **-0.3** as the policy explores randomly. By around **episode 1,500** it crosses zero (the policy has learned that calling at least one relevant specialist reliably beats the generalist baseline). From **episode 8,000 onwards** the smoothed reward stabilizes around **2.0**, where it holds through the end of the run. By the final logged steps, the policy had been operating in **Phase 3 for over 10,000 consecutive episodes**, with a rolling mean oscillating between **1.8 and 2.1** on the hardest task tier. Peak episode rewards reach as high as **+3.87**.

All training checkpoints (at steps 5,000, 10,000, 15,000, 20,000, 25,000, and 30,000) are publicly available on HuggingFace along with the full training logs: [https://huggingface.co/garvitsachdeva/spindleflow-rl/commits/main](https://huggingface.co/garvitsachdeva/spindleflow-rl/commits/main)

| Metric | Value |
|---|---|
| Algorithm | RecurrentPPO (SB3 + sb3-contrib) |
| Policy | LSTM, hidden size 256 |
| Observation space | 5,490-dim flat vector |
| Action space | 12-dim continuous Box |
| Specialists (seed roster) | 8 |
| Total training steps | 30,000 |
| Episodes completed | ~13,400 |
| Curriculum phases completed | 3 / 3 |
| Rolling mean reward (Phase 3) | 1.8 - 2.1 |
| Peak episode reward | +3.87 |

What excites us most is where this goes next. The architecture is built to scale. Extending the run to 500,000 steps would let the Phase 3 policy consolidate further, give the SpawnMemory enough observations to condition specialist design on rich historical patterns, and push the conflict resolution bandit past its exploration phase into confident, consistent exploitation. A larger specialist roster would emerge organically through auto-spawning as the policy encounters more diverse tasks. The environment is ready for all of it. We simply ran as far as the hackathon window allowed, and we're proud of how far that turned out to be.

---

## The Demo Moment

Take the task: *"Design a microservices authentication system with JWT, OAuth2, and rate limiting."*

A generic GPT-4o-mini call gives you five numbered bullet points. Gather requirements. Research patterns. Draft architecture. Implement. Deploy. It's correct the way a textbook is correct -- it covers everything and illuminates nothing.

The trained policy routes this to the security analyst, the backend API engineer, the database architect, and the DevOps engineer. Each specialist builds on the prior analysis via sequential delegation. The generalist baseline scores -0.10 in episode reward. The specialist-routed output scores +0.11, a gap of +0.21 on the same tiered reward signal.

The orchestrator learned to produce that gap.

---

## The Bigger Picture

What we built is not a chatbot wrapper. It's a training environment for orchestration intelligence, a system that makes the conductor smarter over time without anyone rewriting routing logic.

The architecture supports any sector. The specialists are not hardcoded. The policy generalizes to new agents via embeddings. The conflicts resolve themselves, and the resolution strategy improves. New specialists spawn when they're needed and persist in memory for future episodes.

We think this is what enterprise multi-agent systems will look like when they grow up: not static pipelines, but learned delegation strategies that adapt to their roster, their tasks, and their constraints.

SpindleFlow RL is that environment.