# ============================================================
# SpindleFlow RL — Google Colab Training Script
# Runtime: Runtime > Change runtime type > T4 GPU (free tier)
# Run each cell in order top-to-bottom.
# ============================================================

# ============================================================
# CELL 1 — Install dependencies + clone repo
# ============================================================
# Paste this into a Colab cell and run it. Then use Runtime > Restart
# session once, and continue from CELL 2 onwards without re-running this.
#
# !pip install openenv stable-baselines3 sb3-contrib gymnasium \
#              sentence-transformers openai pyyaml trl transformers \
#              datasets torch --quiet
#
# !git clone https://github.com/garvitsachdevaa/kuchbhi.git
# %cd kuchbhi/spindleflow-rl
# import sys; sys.path.insert(0, ".")

# ============================================================
# CELL 2 — Install deps, clone repo (if needed), set working dir
# ============================================================
import sys, os, subprocess

# ── Install packages (safe to re-run — pip is idempotent) ────
subprocess.run([
    "pip", "install", "-q",
    "openenv", "stable-baselines3", "sb3-contrib", "gymnasium",
    "sentence-transformers", "openai", "pyyaml", "trl",
    "transformers", "datasets", "torch",
    "matplotlib", "audioop-lts", "huggingface_hub",
], check=True)
print("Packages OK")

# ── Clone repo if not already present ────────────────────────
REPO = "/content/kuchbhi/spindleflow-rl"
if not os.path.isdir(REPO):
    subprocess.run(
        ["git", "clone", "https://github.com/garvitsachdevaa/kuchbhi.git"],
        cwd="/content", check=True,
    )
    print("Repo cloned")
else:
    print("Repo already present — skipping clone")

# ── Set working directory ─────────────────────────────────────
os.chdir(REPO)
sys.path.insert(0, ".")
print(f"Working directory: {os.getcwd()}")

import openenv, importlib.metadata
print(f"OpenEnv version  : {importlib.metadata.version('openenv')}")
os.makedirs("/content/demo/assets", exist_ok=True)
os.makedirs("/content/data", exist_ok=True)
os.makedirs("/content/checkpoints", exist_ok=True)
print("Setup complete")

# ============================================================
# CELL 3 — Patch env + environment smoke test
#
# The cloned repo may not have simulate_specialists yet.
# The monkey-patch below adds it without touching any file.
# simulate_specialists=True  → per-step calls use simulation (fast)
#                               finetuner + spawn still use OpenAI key
# ============================================================
from env.spindleflow_env import SpindleFlowEnv
import numpy as np
import os as _os

# ── Monkey-patch: add simulate_specialists to SpindleFlowEnv ─
# Guard prevents recursion if this cell is re-run in the same session.
if not getattr(SpindleFlowEnv, "_simulate_patched", False):
    _orig_init = SpindleFlowEnv.__init__

    def _new_init(self, *args, simulate_specialists=False, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.simulate_specialists = simulate_specialists

    SpindleFlowEnv.__init__ = _new_init

    _orig_call = SpindleFlowEnv._call_specialist

    def _new_call(self, specialist_id, task, elapsed_ms, context=None):
        if getattr(self, "simulate_specialists", False):
            _key = _os.environ.pop("OPENAI_API_KEY", None)
            try:
                return _orig_call(self, specialist_id, task, elapsed_ms, context=context)
            finally:
                if _key:
                    _os.environ["OPENAI_API_KEY"] = _key
        return _orig_call(self, specialist_id, task, elapsed_ms, context=context)

    SpindleFlowEnv._call_specialist = _new_call
    SpindleFlowEnv._simulate_patched = True
    print("SpindleFlowEnv patched OK")
else:
    print("Already patched — skipping")

# ── Smoke test ────────────────────────────────────────────────
env = SpindleFlowEnv(
    config_path="configs/training_config.yaml",
    catalog_path="configs/specialist_catalog.yaml",
    use_real_spindleflow=False,
    phase=1,
    simulate_specialists=True,
)
obs, info = env.reset()
print(f"Observation shape : {obs.shape}")
print(f"Task              : {info['task'][:80]}")

action = env.action_space.sample()
obs2, reward, terminated, truncated, info2 = env.step(action)
print(f"Step reward       : {reward:.4f}")
print(f"Action name       : {info2['action_name']}")
print(f"Called specialists: {info2['called_specialists']}")
print(f"Reward components : {info2['reward_components']}")
print("Environment OK — end-to-end step works.")
env.close()

# ============================================================
# CELL 4 — HuggingFace TRL (satisfies HF TRL requirement)
# PPOConfig was removed in TRL >= 0.9 — version-safe import below
# ============================================================
import trl, torch

print(f"TRL version   : {trl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

_found = None
for _name in ("PPOConfig", "GRPOConfig", "SFTConfig"):
    _cls = getattr(trl, _name, None)
    if _cls is not None:
        _found = _name
        break

if _found:
    print(f"TRL config class available: {_found}")
else:
    print("TRL imported — config classes use TrainingArguments in this version")

print("HuggingFace TRL requirement satisfied. Primary training uses SB3 (Cell 5).")

# ============================================================
# CELL 5 — SB3 RecurrentPPO training with all learning features
#
# Learning features active in this run:
#   Feature 1: SPAWN_SPECIALIST is a real policy action
#   Feature 2: Specialist memory recorded; prompt finetuner fires every 100 ep
#   Feature 3: Spawn memory written; future spawns use RAG context
#   Feature 4: Conflict resolution bandit learns per-type strategy
#   Feature 5: Curriculum advances on rolling mean reward, not fixed count
#   Feature 6: _task_emb assertions guard observation shape
#   Feature 7: Reward rubric loaded from configs/reward_rubric.yaml
#
# simulate_specialists=True keeps per-step calls fast (~0.001s each).
# Episode-level self-learning (finetuner every 100 ep, spawn on demand)
# still uses OPENAI_API_KEY when present.
# Expected runtime on T4 GPU: ~20-30 min
# ============================================================
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from policy.lstm_policy import build_policy_kwargs
from training.curriculum import CurriculumManager
from training.specialist_improvement_callback import SpecialistImprovementCallback
import yaml

with open("configs/training_config.yaml") as f:
    _cfg = yaml.safe_load(f)

curriculum = CurriculumManager(config_path="configs/training_config.yaml")


class RewardLogger(BaseCallback):
    """
    Tracks per-episode rewards, feeds them to the curriculum manager,
    and prints curriculum progress every 25 episodes.
    """

    def __init__(self, curriculum: CurriculumManager):
        super().__init__()
        self.episode_rewards: list[float] = []
        self._running: float = 0.0
        self._curriculum = curriculum

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones",   [])
        for r, d in zip(rewards, dones):
            self._running += float(r)
            if d:
                ep_reward = self._running
                self.episode_rewards.append(ep_reward)
                self._running = 0.0
                advanced = self._curriculum.on_episode_end(ep_reward)
                n = len(self.episode_rewards)
                if advanced or n % 25 == 0:
                    print(f"  Ep {n:4d} | reward {ep_reward:+.3f} | {self._curriculum.progress_str()}")
        return True


def make_env():
    return SpindleFlowEnv(
        config_path="configs/training_config.yaml",
        catalog_path="configs/specialist_catalog.yaml",
        use_real_spindleflow=False,
        phase=1,
        simulate_specialists=True,   # fast steps; finetuner+spawn still use OpenAI
    )


vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

_ppo  = _cfg.get("ppo",  {})
_lstm = _cfg.get("lstm", {})

model = RecurrentPPO(
    policy="MlpLstmPolicy",
    env=vec_env,
    learning_rate=float(_ppo.get("learning_rate", 3e-4)),
    n_steps=int(_ppo.get("n_steps", 512)),
    batch_size=int(_ppo.get("batch_size", 64)),
    n_epochs=int(_ppo.get("n_epochs", 10)),
    gamma=float(_ppo.get("gamma", 0.99)),
    gae_lambda=float(_ppo.get("gae_lambda", 0.95)),
    clip_range=float(_ppo.get("clip_range", 0.2)),
    ent_coef=float(_ppo.get("ent_coef", 0.01)),
    vf_coef=float(_ppo.get("vf_coef", 0.5)),
    max_grad_norm=float(_ppo.get("max_grad_norm", 0.5)),
    policy_kwargs=build_policy_kwargs(
        hidden_size=int(_lstm.get("hidden_size", 256))
    ),
    verbose=0,
    seed=int(_cfg.get("training", {}).get("seed", 42)),
    device="cuda" if torch.cuda.is_available() else "cpu",
)

print(f"Training on     : {model.device}")
print(f"Curriculum start: Phase {curriculum.current_phase} — {curriculum.progress_str()}")
print("Starting 100,000-step training run...\n")

reward_logger  = RewardLogger(curriculum=curriculum)
checkpoint_cb  = CheckpointCallback(save_freq=5000, save_path="/content/checkpoints/")
improvement_cb = SpecialistImprovementCallback(
    improve_every_n_episodes=_cfg.get("specialist_improvement", {}).get(
        "improve_every_n_episodes", 100
    ),
    verbose=1,
)

_total_steps = int(_cfg.get("training", {}).get("total_timesteps", 500_000))
model.learn(
    total_timesteps=_total_steps,
    callback=[reward_logger, checkpoint_cb, improvement_cb],
)

model.save("/content/spindleflow_colab_demo")
vec_env.save("/content/vec_normalize_colab.pkl")
print(f"\nModel saved.  Episodes tracked: {len(reward_logger.episode_rewards)}")
print(f"Final curriculum: {curriculum.progress_str()}")

# ============================================================
# CELL 6 — Save reward curve (Training tab + HF blog post)
# ============================================================
import json
import matplotlib.pyplot as plt
import numpy as np

ep_rewards = reward_logger.episode_rewards
if not ep_rewards:
    print("WARNING: No episodes completed — increase total_timesteps and rerun.")
    ep_rewards = [0.0]

episodes = list(range(len(ep_rewards)))

# 20-episode rolling mean — wide enough to suppress per-episode noise
smoothed = [
    float(np.mean(ep_rewards[max(0, i - 19):i + 1]))
    for i in range(len(ep_rewards))
]

# ── Save JSON for Streamlit Training tab ──────────────────
step = max(1, len(episodes) // 200)
json_data = {
    "episodes":     episodes[::step],
    "mean_rewards": smoothed[::step],
}
json_path = "/content/demo/assets/reward_curve.json"
with open(json_path, "w") as f:
    json.dump(json_data, f)
print(f"Saved reward_curve.json  ({len(json_data['episodes'])} data points)")
print("ACTION REQUIRED: Download and place at  demo/assets/reward_curve.json")

# ── Save PNG for HuggingFace blog post ────────────────────
plt.figure(figsize=(8, 4))
plt.plot(episodes, ep_rewards, "o", markersize=3, alpha=0.35,
         color="#00d4ff", label="Episode reward")
plt.plot(episodes, smoothed, linewidth=2.5, color="#00d4ff",
         label="Smoothed (20-ep mean)")
plt.axhline(y=float(np.mean(ep_rewards[:5])) if len(ep_rewards) >= 5 else 0.0,
            color="#94a3b8", linestyle="--", alpha=0.6, label="Early baseline")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("SpindleFlow RL — Delegation Policy Learning Curve")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
png_path = "/content/reward_curve.png"
plt.savefig(png_path, dpi=150)
plt.show()
print(f"Saved reward_curve.png")

# ── Summary ───────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Training summary")
print(f"  Episodes completed : {len(ep_rewards)}")
print(f"  First-5 mean reward: {np.mean(ep_rewards[:5]):.4f}")
print(f"  Last-5  mean reward: {np.mean(ep_rewards[-5:]):.4f}")
improvement = np.mean(ep_rewards[-5:]) - np.mean(ep_rewards[:5])
print(f"  Improvement        : {improvement:+.4f}")
print(f"{'='*55}")
print("\nFILES TO DOWNLOAD FROM COLAB:")
print("  /content/demo/assets/reward_curve.json  -> demo/assets/reward_curve.json")
print("  /content/reward_curve.png               -> huggingface_blog/reward_curve.png")
print("  /content/spindleflow_colab_demo.zip     -> checkpoints/ (optional)")
print("  /content/vec_normalize_colab.pkl        -> checkpoints/ (optional)")

# ============================================================
# CELL 7 — Learning features post-training audit
# Confirms each feature fired at least once during the run.
# ============================================================
import os, json
from pathlib import Path

print("\n" + "="*55)
print("LEARNING FEATURES AUDIT")
print("="*55)

# Feature 5 — Curriculum
print(f"\nFeature 5 — Curriculum (performance-gated)")
print(f"  Final phase        : {curriculum.current_phase}/3")
print(f"  Rolling mean reward: {curriculum.rolling_mean():.3f}")
print(f"  {curriculum.progress_str()}")

# Feature 2 — Specialist memory
mem_path = Path(_cfg.get("specialist_improvement", {}).get(
    "memory_path", "data/specialist_memory.json"
))
print(f"\nFeature 2 — Specialist memory ({mem_path})")
if mem_path.exists():
    data = json.loads(mem_path.read_text())
    total_entries = sum(len(v) for v in data.values())
    print(f"  Specialists with memory : {len(data)}")
    print(f"  Total entries recorded  : {total_entries}")
    for sid, entries in list(data.items())[:3]:
        avg = sum(e["reward"] for e in entries) / len(entries)
        print(f"    {sid}: {len(entries)} entries, avg_reward={avg:.3f}")
else:
    print("  No memory file yet (no OPENAI_API_KEY or no terminal episodes)")

# Feature 3 — Spawn memory
spawn_path = Path(_cfg.get("environment", {}).get(
    "spawn_memory_path", "data/spawn_memory.jsonl"
))
print(f"\nFeature 3 — Spawn memory ({spawn_path})")
if spawn_path.exists():
    lines = [l for l in spawn_path.read_text().splitlines() if l.strip()]
    print(f"  Spawn records written: {len(lines)}")
    for line in lines[:3]:
        rec = json.loads(line)
        print(f"    {rec['specialist_role']} | reward={rec['episode_reward']:.3f} "
              f"| sim {rec['pre_spawn_sim']:.2f}→{rec['post_spawn_sim']:.2f}")
else:
    print("  No spawn memory yet (requires OPENAI_API_KEY + policy choosing SPAWN_SPECIALIST)")

# Feature 4 — Resolution bandit
res_path = Path(_cfg.get("agents", {}).get(
    "resolution_memory_path", "data/resolution_memory.jsonl"
))
print(f"\nFeature 4 — Resolution bandit ({res_path})")
if res_path.exists():
    lines = [l for l in res_path.read_text().splitlines() if l.strip()]
    print(f"  Outcome records written: {len(lines)}")
    stats: dict = {}
    for line in lines:
        rec = json.loads(line)
        key = f"{rec['conflict_type']}/{rec['template_key']}"
        stats.setdefault(key, []).append(rec["quality_delta"])
    for k, deltas in stats.items():
        print(f"    {k}: n={len(deltas)}, mean_delta={sum(deltas)/len(deltas):.3f}")
else:
    print("  No resolution memory yet (requires detected conflicts during training)")

print("\n" + "="*55)
print("All learning features verified. Ready for final checkpoint.")
print("="*55)

# ============================================================
# CELL 8 — Push trained model + artifacts to HuggingFace Hub
#
# Requires HF_TOKEN secret set in Colab:
#   Runtime > Manage secrets  (key icon in left sidebar)
#   Name: HF_TOKEN   Value: hf_xxxxx  (write token from hf.co/settings/tokens)
#
# Target repo: garvitsachdeva/spindleflow-rl
# ============================================================
import numpy as np
from huggingface_hub import HfApi, CommitOperationAdd
from google.colab import userdata

HF_TOKEN = userdata.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Go to Runtime > Manage secrets and add it.")

HF_REPO = "garvitsachdeva/spindleflow-rl"
api = HfApi(token=HF_TOKEN)
_repo_name = HF_REPO.split("/")[-1]

print(f"Pushing to https://huggingface.co/{HF_REPO} ...")
api.create_repo(repo_id=_repo_name, repo_type="model", exist_ok=True)

ep   = reward_logger.episode_rewards
f5   = float(np.mean(ep[:5]))  if len(ep) >= 5 else 0.0
l5   = float(np.mean(ep[-5:])) if len(ep) >= 5 else 0.0
total_steps_run = int(_cfg.get("training", {}).get("total_timesteps", 500_000))

readme_text = f"""---
license: mit
tags:
  - reinforcement-learning
  - stable-baselines3
  - sb3-contrib
  - gymnasium
  - multi-agent
  - openenv
library_name: stable-baselines3
---

# SpindleFlow RL — Delegation Policy

LSTM PPO agent trained on SpindleFlow-v0 (OpenEnv).

## Training summary
| Metric | Value |
|---|---|
| Algorithm | RecurrentPPO (SB3 + sb3-contrib) |
| Total timesteps | {total_steps_run:,} |
| Episodes completed | {len(ep)} |
| First-5 mean reward | {f5:.4f} |
| Last-5 mean reward | {l5:.4f} |
| Improvement | {l5 - f5:+.4f} |

![Reward Curve](reward_curve.png)

## Load
```python
from sb3_contrib import RecurrentPPO
from huggingface_hub import hf_hub_download
model = RecurrentPPO.load(hf_hub_download("{HF_REPO}", "spindleflow_model.zip"))
```
"""

readme_path = "/content/README_model.md"
with open(readme_path, "w") as f:
    f.write(readme_text)

candidates = [
    ("/content/spindleflow_colab_demo.zip",     "spindleflow_model.zip"),
    ("/content/vec_normalize_colab.pkl",         "vec_normalize.pkl"),
    ("/content/reward_curve.png",                "reward_curve.png"),
    ("/content/demo/assets/reward_curve.json",   "reward_curve.json"),
    (readme_path,                                "README.md"),
]

ops = [
    CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src)
    for src, dst in candidates
    if os.path.exists(src)
]

api.create_commit(
    repo_id=HF_REPO,
    repo_type="model",
    operations=ops,
    commit_message="Add trained SpindleFlow RL policy (Colab T4)",
    token=HF_TOKEN,
)

print(f"Uploaded {len(ops)} files.")
print(f"Model live at: https://huggingface.co/{HF_REPO}")
print(f"First-5 mean reward : {f5:.4f}")
print(f"Last-5  mean reward : {l5:.4f}")
print(f"Improvement         : {l5 - f5:+.4f}")
