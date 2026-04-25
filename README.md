---
title: SpindleFlow RL
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.40.0"
app_file: app.py
pinned: false
---

# SpindleFlow RL — Delegation Policy RL Environment

An RL environment that trains an orchestrator to **learn** delegation strategy,
built on top of the SpindleFlow multi-agent execution system.

## Architecture

```
SpindleFlow (TypeScript) ← execution backend
SpindleFlow RL (Python)  ← RL training layer
```

The RL agent learns *which specialists to call, in what mode, and when to stop* —
not how to write YAML. SpindleFlow executes the decisions; the RL policy makes them.

## Key Design Decisions

| Component | Design | Why |
|---|---|---|
| Reward | Tiered cascade (0/1/2/3) with episode-level tier lock | Valid delta, no tier drift, $8/1000-episode run |
| Roster | Capability embeddings (all-MiniLM-L6-v2, 384-dim) | Zero-shot generalization to new specialists |
| Delegation | DAG with cycle detection + action masking | No A→B→A loops |
| Policy | LSTM PPO (RecurrentPPO, SB3) | POMDP-safe for scratchpad context |
| Graph encoding | Padded adjacency MLP (not GNN) | Hackathon-feasible; GNN for production |
| Consistency | Dirichlet prior (alpha=1.0) | Non-zero reward from Episode 1 |
| Stopping | STOP as explicit learned action (Head 1) | Adaptive, not hardcoded |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install sb3-contrib

# 2. Set environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Run smoke tests
pytest tests/ -v

# 4. Pre-compute demo assets
python demo/precompute_demo.py

# 5. Start training (Phase 1)
python training/train.py --phase 1 --timesteps 50000

# 6. Watch training curves
tensorboard --logdir tensorboard_logs/

# 7. Run demo
python demo/run_demo.py
```

## Reward Function

```python
total_reward = (
    quality_delta          # specialist_score - baseline_score (same tier)
  - efficiency_penalty     # 0.05 * max(0, n_specialists - expected)
  - failure_penalty        # 0.3 per timeout, 0.2 per error (reduced if fallback)
  + recovery_bonus         # 0.1 if fallback recovered successfully
  - conflict_penalty       # 0.1 per unresolved conflict
  + conflict_bonus         # 0.05 per resolved conflict
  + consistency_bonus      # 0.1 * Dirichlet-prior path consistency
  - latency_penalty        # latency_weight * overage_fraction (tunable)
  + explanation_bonus      # 0.05 if delegation is auditable
)
```

## Project Structure

```
spindleflow-rl/
├── env/                   ← Gymnasium environment + state/action/graph
├── reward/                ← Tiered reward, failure/conflict/latency signals
├── agents/                ← Task decomposer, fallback chains, conflict resolver
├── policy/                ← LSTM policy, state encoder, action heads
├── training/              ← PPO training loop, curriculum, task bank
├── transfer/              ← Cross-company fine-tuning strategy
├── audit/                 ← Delegation trace + explanation generation
├── security/              ← Scratchpad sandbox isolation
├── demo/                  ← Before/after demo assets + precompute script
├── colab/                 ← Google Colab training notebook
├── huggingface_blog/      ← HuggingFace mini-blog
├── tests/                 ← Pytest test suite (20 tests, all passing)
└── configs/               ← Specialist catalog + training hyperparameters
```

## OpenEnv Compliance

`SpindleFlow-v0` is registered with OpenEnv (hackathon requirement):

```python
import env.openenv_wrapper  # triggers registration
from env.openenv_wrapper import verify_openenv_compliance
verify_openenv_compliance()  # True
```

## Observation Space

Flat `(5490,)` float32 vector (for `max_specialists=6`):

| Component | Dim |
|---|---|
| Task embedding | 384 |
| Roster embeddings (6×384) | 2304 |
| Called embeddings (6×384) | 2304 |
| Scratchpad embedding | 384 |
| Delegation graph adjacency | 100 |
| Called specialist mask | 6 |
| Scalar features | 8 |
| **Total** | **5490** |

## Action Space

Flat `(12,)` continuous Box (for `max_specialists=6`):

| Slot | Meaning |
|---|---|
| `[0]` | Meta-action (CALL_SPECIALIST / STOP / …) |
| `[1:7]` | Specialist selection logits (multi-hot) |
| `[7]` | Delegation mode (SEQUENTIAL / PARALLEL / …) |
| `[8:12]` | Mode parameters (rounds, threshold, budget) |

## Training

```bash
# Demo mode (no OpenAI calls, fast)
python training/train.py --phase 1 --timesteps 50000 --demo-mode

# Full run with T2 reward
python training/train.py --phase 1 --timesteps 100000

# Resume from checkpoint
python training/train.py --checkpoint checkpoints/spindleflow_rl_50000_steps.zip
```

## Colab

See [colab/README_COLAB.md](colab/README_COLAB.md) for Google Colab quick start (T4 GPU, free tier).

## HuggingFace

See [huggingface_blog/blog_post.md](huggingface_blog/blog_post.md) for the submission blog post.
