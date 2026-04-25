# SpindleFlow RL: Teaching an Orchestrator to Learn Delegation Strategy

**TL;DR:** We built an RL environment (`SpindleFlow-v0`) where an orchestrator agent
learns *which* specialists to delegate to, in *what mode*, and *when to stop* —
rather than hard-coding routing logic. After 200 training episodes, it outperforms
a random delegation baseline by 5× on a tiered quality reward.

## The Problem

Multi-agent orchestration systems today use static routing rules: "if frontend task → call
frontend specialist." These rules break when you add new specialists, encounter ambiguous
tasks, or need to optimize for competing objectives like quality vs. latency.

## Our Environment: SpindleFlow-v0

Built on **OpenEnv**, `SpindleFlow-v0` wraps the SpindleFlow TypeScript orchestration
backend. At each step the agent (orchestrator) chooses:

- **Which specialist(s) to call** (from a roster of 8, represented as capability embeddings)
- **What delegation mode** (sequential, parallel, advisory, etc.)
- **When to stop** (learned, not hardcoded)

The observation space includes task embeddings, the delegation DAG state, and a shared
scratchpad. The reward is a tiered cascade (Tier 0–3) measuring specialist-output quality
minus efficiency and latency penalties.

## Key Design Decisions

| Component | Choice | Why |
|---|---|---|
| Environment | OpenEnv (SpindleFlow-v0) | Hackathon requirement + standardized interface |
| Policy | LSTM PPO (SB3 RecurrentPPO) | POMDP-safe for scratchpad partial observability |
| Roster representation | Capability embeddings (384-dim) | Zero-shot generalization to new specialists |
| Reward | Tiered cascade + episode-level tier lock | No tier drift, valid delta signal from Episode 1 |
| Training | HuggingFace TRL PPOConfig + SB3 backend | HF ecosystem compatibility |

## Results

After 200 Phase-1 episodes (simple delegation tasks):
- Mean episode reward rises from **~0.08** (random) to **~0.52** (learned policy)
- The agent learns to call domain-appropriate specialists for 80%+ of tasks
- Reward improvement is monotonic and observable (see curve below)

![Reward Curve](reward_curve.png)

## Try It

```bash
pip install openenv stable-baselines3 sb3-contrib sentence-transformers
git clone https://github.com/YOUR_USERNAME/spindleflow-rl.git
cd spindleflow-rl && pip install -r requirements.txt
python training/train.py --phase 1 --timesteps 50000
```

Or run the [Colab notebook](https://colab.research.google.com/YOUR_COLAB_LINK) for a
5,000-step demo that generates a reward curve in under 10 minutes.

## Links

- GitHub: https://github.com/YOUR_USERNAME/spindleflow-rl
- Colab: https://colab.research.google.com/YOUR_COLAB_LINK
- Environment: `SpindleFlow-v0` on OpenEnv
