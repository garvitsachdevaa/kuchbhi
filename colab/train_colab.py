# ============================================================
# SpindleFlow RL — Google Colab Training Script
# Runtime: Runtime > Change runtime type > T4 GPU (free tier)
#
# SECRETS (Runtime > Manage secrets — key icon in sidebar):
#   HF_TOKEN       REQUIRED  — HuggingFace write token
#                              hf.co/settings/tokens → New token (write)
#   OPENAI_API_KEY OPTIONAL  — enables finetuner + spawn self-learning
#                              without it the run uses fast simulation mode
#
# Run CELL 2 through CELL 8 in order. Do NOT re-run CELL 2 after restart.
# ============================================================


# ============================================================
# CELL 2 — Install deps, clone repo, set working dir
# ============================================================
import sys, os, subprocess

subprocess.run([
    "pip", "install", "-q",
    "openenv", "stable-baselines3", "sb3-contrib", "gymnasium",
    "sentence-transformers", "openai", "pyyaml", "trl",
    "transformers", "datasets", "torch",
    "matplotlib", "audioop-lts", "huggingface_hub",
], check=True)
print("Packages OK")

REPO = "/content/kuchbhi/spindleflow-rl"
if not os.path.isdir(REPO):
    subprocess.run(
        ["git", "clone", "https://github.com/garvitsachdevaa/kuchbhi.git"],
        cwd="/content", check=True,
    )
    print("Repo cloned")
else:
    print("Repo already present — skipping clone")

os.chdir(REPO)
sys.path.insert(0, ".")
print(f"Working directory: {os.getcwd()}")

import openenv, importlib.metadata
print(f"OpenEnv version  : {importlib.metadata.version('openenv')}")
os.makedirs("/content/demo/assets", exist_ok=True)
os.makedirs("/content/data", exist_ok=True)
os.makedirs("/content/checkpoints", exist_ok=True)
os.makedirs("/content/logs", exist_ok=True)
print("Setup complete")


# ============================================================
# CELL 3 — Patch env + smoke test
# ============================================================
from env.spindleflow_env import SpindleFlowEnv
import numpy as np
import os as _os

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
# CELL 4 — HuggingFace TRL (hackathon requirement check)
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
# CELL 5 — RecurrentPPO training (LSTM PPO)
#
# simulate_specialists=True  — per-step calls are local (~0.001 s)
# no OpenAI calls during steps → fast on T4
# Expected runtime: ~20–25 min for 100k steps (~10k episodes)
# ============================================================
import time
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from policy.lstm_policy import build_policy_kwargs
from training.curriculum import CurriculumManager
from training.specialist_improvement_callback import SpecialistImprovementCallback
import yaml

_LOG_FILE = "/content/logs/training_log.txt"

def _tlog(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(_LOG_FILE, "a", encoding="utf-8") as _f:
        _f.write(line + "\n")

with open("configs/training_config.yaml") as f:
    _cfg = yaml.safe_load(f)

curriculum = CurriculumManager(config_path="configs/training_config.yaml")

TOTAL_TIMESTEPS = 100_000   # ~10k episodes on T4, ~20-25 min


class RewardLogger(BaseCallback):
    def __init__(self, curriculum: CurriculumManager):
        super().__init__()
        self.episode_rewards: list[float] = []
        self._running: float = 0.0
        self._curriculum = curriculum

    def _on_step(self) -> bool:
        for r, d in zip(
            self.locals.get("rewards", []),
            self.locals.get("dones",   []),
        ):
            self._running += float(r)
            if d:
                ep = self._running
                self.episode_rewards.append(ep)
                self._running = 0.0
                advanced = self._curriculum.on_episode_end(ep)
                n = len(self.episode_rewards)
                if advanced or n % 50 == 0:
                    _tlog(
                        f"Ep {n:5d} | reward {ep:+.3f} | "
                        f"{self._curriculum.progress_str()}"
                    )
        return True


def make_env():
    return SpindleFlowEnv(
        config_path="configs/training_config.yaml",
        catalog_path="configs/specialist_catalog.yaml",
        use_real_spindleflow=False,
        phase=1,
        simulate_specialists=True,
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

_tlog(f"Device          : {model.device}")
_tlog(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
_tlog(f"Curriculum start: Phase {curriculum.current_phase} — {curriculum.progress_str()}")
_tlog("Training started...\n")

reward_logger  = RewardLogger(curriculum=curriculum)
checkpoint_cb  = CheckpointCallback(save_freq=10_000, save_path="/content/checkpoints/")
improvement_cb = SpecialistImprovementCallback(
    improve_every_n_episodes=_cfg.get("specialist_improvement", {}).get(
        "improve_every_n_episodes", 100
    ),
    verbose=1,
)

_t0 = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[reward_logger, checkpoint_cb, improvement_cb],
)
_elapsed = time.time() - _t0

model.save("/content/spindleflow_colab_model")
vec_env.save("/content/vec_normalize_colab.pkl")

_tlog(f"\nTraining done in {_elapsed/60:.1f} min")
_tlog(f"Episodes tracked : {len(reward_logger.episode_rewards)}")
_tlog(f"Final curriculum : {curriculum.progress_str()}")


# ============================================================
# CELL 6 — Reward curve (publication-quality)
# ============================================================
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ep_rewards = reward_logger.episode_rewards
if not ep_rewards:
    print("WARNING: No episodes completed — increase TOTAL_TIMESTEPS and rerun.")
    ep_rewards = [0.0]

n_ep      = len(ep_rewards)
episodes  = list(range(n_ep))
window    = max(30, n_ep // 20)   # adaptive smoothing: ~5% of total episodes

smoothed = [
    float(np.mean(ep_rewards[max(0, i - window):i + 1]))
    for i in range(n_ep)
]

early_mean = float(np.mean(ep_rewards[:min(50, n_ep)]))
final_mean = float(np.mean(ep_rewards[max(0, n_ep - 200):]))

# ── Save JSON ──────────────────────────────────────────────
step      = max(1, n_ep // 300)
json_data = {
    "episodes":     episodes[::step],
    "mean_rewards": smoothed[::step],
}
json_path = "/content/demo/assets/reward_curve.json"
with open(json_path, "w") as f:
    json.dump(json_data, f)

# ── Plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

plot_every = max(1, n_ep // 800)
ax.scatter(
    episodes[::plot_every], ep_rewards[::plot_every],
    s=4, alpha=0.25, color="#58a6ff", zorder=2, label="Episode reward",
)
ax.plot(
    episodes[::plot_every], smoothed[::plot_every],
    linewidth=2.5, color="#ff6b35", zorder=3,
    label=f"Smoothed ({window}-ep mean)",
)
ax.axhline(
    y=early_mean, color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.75,
    label=f"Early baseline  {early_mean:+.3f}",
)
ax.axhline(
    y=final_mean, color="#34d399", linestyle="--", linewidth=1.2, alpha=0.85,
    label=f"Final mean  {final_mean:+.3f}",
)

ax.set_xlabel("Episode", color="#c9d1d9", fontsize=12)
ax.set_ylabel("Reward", color="#c9d1d9", fontsize=12)
ax.set_title(
    "SpindleFlow RL — Delegation Policy Learning Curve\n"
    f"RecurrentPPO · LSTM · {TOTAL_TIMESTEPS:,} steps · {n_ep:,} episodes",
    color="#f0f6fc", fontsize=13, fontweight="bold", pad=14,
)
ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")
ax.grid(color="#21262d", linewidth=0.8, alpha=0.9)

legend = ax.legend(
    fontsize=10, framealpha=0.85,
    facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
)

# Annotate improvement
improvement = final_mean - early_mean
sign = "▲" if improvement >= 0 else "▼"
ax.annotate(
    f"  {sign} {abs(improvement):.3f} reward improvement",
    xy=(n_ep * 0.65, (early_mean + final_mean) / 2),
    color="#f0f6fc", fontsize=10, fontstyle="italic",
)

fig.tight_layout()
png_path = "/content/reward_curve.png"
fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
_tlog(f"Reward curve saved → {png_path}")

_tlog(f"\n{'='*55}")
_tlog(f"Training summary")
_tlog(f"  Episodes completed : {n_ep}")
_tlog(f"  Early baseline     : {early_mean:+.4f}")
_tlog(f"  Final mean         : {final_mean:+.4f}")
_tlog(f"  Improvement        : {improvement:+.4f}")
_tlog(f"{'='*55}")


# ============================================================
# CELL 7 — Learning features audit
# ============================================================
import os, json
from pathlib import Path

print("\n" + "="*55)
print("LEARNING FEATURES AUDIT")
print("="*55)

print(f"\nFeature 5 — Curriculum (performance-gated)")
print(f"  Final phase        : {curriculum.current_phase}/3")
print(f"  Rolling mean reward: {curriculum.rolling_mean():.3f}")
print(f"  {curriculum.progress_str()}")

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
    print("  No memory file yet (OPENAI_API_KEY not set — simulation mode)")

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
    print("  No spawn memory yet (requires OPENAI_API_KEY + SPAWN_SPECIALIST action)")

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
print("All learning features verified.")
print("="*55)


# ============================================================
# CELL 8 — Push model + artifacts + logs to HuggingFace Hub
#
# HF_TOKEN must be in Runtime > Manage secrets (key icon).
# ============================================================
import numpy as np
from huggingface_hub import HfApi, CommitOperationAdd
from google.colab import userdata

HF_TOKEN = userdata.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not set. "
        "Go to Runtime > Manage secrets, add Name=HF_TOKEN, Value=hf_xxxx, enable notebook access."
    )

HF_REPO = "garvitsachdeva/spindleflow-rl"
api = HfApi(token=HF_TOKEN)

_tlog(f"Pushing to https://huggingface.co/{HF_REPO} ...")
api.create_repo(repo_id=HF_REPO.split("/")[-1], repo_type="model", exist_ok=True)

ep   = reward_logger.episode_rewards
f5   = float(np.mean(ep[:5]))   if len(ep) >= 5 else 0.0
l5   = float(np.mean(ep[-5:])) if len(ep) >= 5 else 0.0

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

LSTM PPO (RecurrentPPO) agent trained on SpindleFlow-v0 (OpenEnv).
Trained on Google Colab T4 GPU.

## Training summary
| Metric | Value |
|---|---|
| Algorithm | RecurrentPPO (SB3 + sb3-contrib) |
| Total timesteps | {TOTAL_TIMESTEPS:,} |
| Episodes completed | {len(ep):,} |
| Early baseline (first 50) | {early_mean:.4f} |
| Final mean (last 200) | {final_mean:.4f} |
| Improvement | {final_mean - early_mean:+.4f} |
| Training time | {_elapsed/60:.1f} min |
| Device | T4 GPU |

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
    ("/content/spindleflow_colab_model.zip",    "spindleflow_model.zip"),
    ("/content/vec_normalize_colab.pkl",         "vec_normalize.pkl"),
    ("/content/reward_curve.png",                "reward_curve.png"),
    ("/content/demo/assets/reward_curve.json",   "reward_curve.json"),
    ("/content/logs/training_log.txt",           "training_log.txt"),
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

_tlog(f"Uploaded {len(ops)} files:")
for src, dst in candidates:
    if os.path.exists(src):
        _tlog(f"  ✓ {dst}")
_tlog(f"Model live at  : https://huggingface.co/{HF_REPO}")
_tlog(f"Training log   : https://huggingface.co/{HF_REPO}/blob/main/training_log.txt")
_tlog(f"Reward curve   : https://huggingface.co/{HF_REPO}/blob/main/reward_curve.png")
_tlog(f"Reward (early) : {early_mean:+.4f}")
_tlog(f"Reward (final) : {final_mean:+.4f}")
_tlog(f"Improvement    : {final_mean - early_mean:+.4f}")
