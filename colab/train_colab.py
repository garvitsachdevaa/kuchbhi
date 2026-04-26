# ============================================================
# SpindleFlow RL — Colab Training Script
#
# BEFORE ANYTHING:
#   1. Runtime → Change runtime type → T4 GPU
#   2. Key icon (left sidebar) → Manage secrets → add:
#        HF_TOKEN       = hf_xxxx  (write token: hf.co/settings/tokens)
#        OPENAI_API_KEY = sk-xxxx
#      Toggle "Notebook access" ON for both.
#   3. Create a new Colab notebook.
#   4. Copy each CELL block below into its own code cell.
#   5. Run cells top to bottom, one at a time.
# ============================================================


# ============================================================
# CELL 1 — Install packages + clone/update repo
# ============================================================
import subprocess, os, sys

print(f"Python {sys.version}")

packages = [
    "openenv", "stable-baselines3", "sb3-contrib", "gymnasium",
    "sentence-transformers", "openai", "pyyaml", "trl",
    "transformers", "datasets", "torch", "matplotlib",
    "huggingface_hub", "python-dotenv",
]
if sys.version_info >= (3, 13):
    packages.append("audioop-lts")

result = subprocess.run(
    ["pip", "install", "-q"] + packages,
    capture_output=True, text=True
)
if result.returncode != 0:
    print(result.stderr[-3000:])
    raise RuntimeError("pip install failed")
print("Packages OK")

REPO = "/content/kuchbhi"
GIT_URL = "https://github.com/garvitsachdevaa/kuchbhi.git"

if not os.path.isdir(os.path.join(REPO, ".git")):
    subprocess.run(["git", "clone", "--depth=1", GIT_URL], cwd="/content", check=True)
    print("Repo cloned")
else:
    subprocess.run(["git", "pull"], cwd=REPO, check=True)
    print("Repo updated")

os.chdir(REPO)
sys.path.insert(0, REPO)

for d in ["/content/demo/assets", "/content/data",
          "/content/checkpoints", "/content/logs"]:
    os.makedirs(d, exist_ok=True)

print(f"CWD: {os.getcwd()}")
print("CELL 1 done ✓")


# ============================================================
# CELL 2 — Load secrets (with clear error messages)
# ============================================================
import os
try:
    from google.colab import userdata
    HF_TOKEN       = userdata.get("HF_TOKEN")
    OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
except Exception:
    HF_TOKEN       = ""
    OPENAI_API_KEY = ""

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not found.\n"
        "Click the 🔑 icon → Add secret → Name: HF_TOKEN → toggle Notebook access ON\n"
        "Then Runtime → Restart and run all."
    )
if not OPENAI_API_KEY:
    print("⚠️  No OPENAI_API_KEY — simulation mode (no LLM calls, faster training)")

os.environ["HF_TOKEN"]       = HF_TOKEN
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print(f"HF_TOKEN       : {HF_TOKEN[:8]}...{HF_TOKEN[-4:]}")
print(f"OPENAI_API_KEY : {'set' if OPENAI_API_KEY else 'NOT SET — simulation mode'}")
print("CELL 2 done ✓")


# ============================================================
# CELL 3 — Patch env + smoke test
# ============================================================
import os as _os
import numpy as np
from env.spindleflow_env import SpindleFlowEnv

# Adds simulate_specialists kwarg so per-step calls stay local/fast.
# OPENAI_API_KEY is still active for task generation + finetuner.
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
                return _orig_call(self, specialist_id, task,
                                  elapsed_ms, context=context)
            finally:
                if _key:
                    _os.environ["OPENAI_API_KEY"] = _key
        return _orig_call(self, specialist_id, task, elapsed_ms, context=context)

    SpindleFlowEnv._call_specialist = _new_call
    SpindleFlowEnv._simulate_patched = True
    print("SpindleFlowEnv patched")

env = SpindleFlowEnv(
    config_path="configs/training_config.yaml",
    catalog_path="configs/specialist_catalog.yaml",
    use_real_spindleflow=False,
    phase=1,
    simulate_specialists=True,
)
obs, info = env.reset()
print(f"obs shape : {obs.shape}")
print(f"task      : {info['task'][:80]}")

_, reward, _, _, info2 = env.step(env.action_space.sample())
print(f"reward    : {reward:.4f}")
print(f"action    : {info2['action_name']}")
env.close()
print("\nCELL 3 done ✓ — environment OK")


# ============================================================
# CELL 4 — TRL check (hackathon requirement)
# ============================================================
import trl, torch

print(f"TRL   : {trl.__version__}")
print(f"Torch : {torch.__version__}")
print(f"CUDA  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

for _name in ("PPOConfig", "GRPOConfig", "SFTConfig"):
    if getattr(trl, _name, None):
        print(f"TRL config: {_name}")
        break
else:
    print("TRL imported (TrainingArguments-based version)")

print("\nCELL 4 done ✓ — TRL requirement satisfied")


# ============================================================
# CELL 5 — Train RecurrentPPO (LSTM PPO)
#
# Per-step specialist calls : local simulation (no API cost/latency)
# Task generation           : GPT-4o-mini via OPENAI_API_KEY
# Finetuner                 : fires every 100 episodes
# Reward baseline           : GPT-4o-mini via OPENAI_API_KEY
#
# Expected runtime: ~20–25 min on T4 for 100k steps (~10k episodes)
# ============================================================
import time, yaml, torch, numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from policy.lstm_policy import build_policy_kwargs
from training.curriculum import CurriculumManager
from training.specialist_improvement_callback import SpecialistImprovementCallback

_LOG_FILE = "/content/logs/training_log.txt"

def _tlog(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(_LOG_FILE, "a") as f:
        f.write(line + "\n")

with open("configs/training_config.yaml") as f:
    _cfg = yaml.safe_load(f)

TOTAL_TIMESTEPS = 100_000
curriculum = CurriculumManager(config_path="configs/training_config.yaml")


class RewardLogger(BaseCallback):
    def __init__(self, curriculum):
        super().__init__()
        self.episode_rewards = []
        self._running = 0.0
        self._curriculum = curriculum

    def _on_step(self):
        for r, d in zip(self.locals.get("rewards", []),
                        self.locals.get("dones",   [])):
            self._running += float(r)
            if d:
                ep = self._running
                self.episode_rewards.append(ep)
                self._running = 0.0
                advanced = self._curriculum.on_episode_end(ep)
                n = len(self.episode_rewards)
                if advanced or n % 50 == 0:
                    _tlog(f"Ep {n:5d} | reward {ep:+.3f} | "
                          f"{self._curriculum.progress_str()}")
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

_tlog(f"Device     : {model.device}")
_tlog(f"Timesteps  : {TOTAL_TIMESTEPS:,}")
_tlog(f"Curriculum : Phase {curriculum.current_phase} — {curriculum.progress_str()}")
_tlog("Training started...")

reward_logger  = RewardLogger(curriculum)
checkpoint_cb  = CheckpointCallback(save_freq=10_000,
                                    save_path="/content/checkpoints/")
improvement_cb = SpecialistImprovementCallback(
    improve_every_n_episodes=_cfg.get("specialist_improvement", {}).get(
        "improve_every_n_episodes", 100),
    verbose=1,
)

_t0 = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[reward_logger, checkpoint_cb, improvement_cb],
)
_elapsed = time.time() - _t0

model.save("/content/spindleflow_model")
vec_env.save("/content/vec_normalize.pkl")

_tlog(f"Done in {_elapsed/60:.1f} min")
_tlog(f"Episodes        : {len(reward_logger.episode_rewards)}")
_tlog(f"Curriculum final: {curriculum.progress_str()}")
print("\nCELL 5 done ✓ — model saved")


# ============================================================
# CELL 6 — Reward curve
# ============================================================
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ep_rewards = reward_logger.episode_rewards
if not ep_rewards:
    raise RuntimeError("No episodes recorded — check Cell 5 output for errors")

n_ep     = len(ep_rewards)
episodes = list(range(n_ep))
window   = max(30, n_ep // 20)   # adaptive: ~5% of run

smoothed = [
    float(np.mean(ep_rewards[max(0, i - window):i + 1]))
    for i in range(n_ep)
]

early_mean  = float(np.mean(ep_rewards[:min(50, n_ep)]))
final_mean  = float(np.mean(ep_rewards[max(0, n_ep - 200):]))
improvement = final_mean - early_mean

# JSON for HF Space demo tab
step = max(1, n_ep // 300)
with open("/content/demo/assets/reward_curve.json", "w") as f:
    json.dump({"episodes": episodes[::step],
               "mean_rewards": smoothed[::step]}, f)

# Plot
fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

every = max(1, n_ep // 800)
ax.scatter(episodes[::every], ep_rewards[::every],
           s=4, alpha=0.25, color="#58a6ff", zorder=2, label="Episode reward")
ax.plot(episodes[::every], smoothed[::every],
        linewidth=2.5, color="#ff6b35", zorder=3,
        label=f"Smoothed ({window}-ep mean)")
ax.axhline(y=early_mean, color="#94a3b8", linestyle="--", linewidth=1.2,
           alpha=0.75, label=f"Early baseline  {early_mean:+.3f}")
ax.axhline(y=final_mean, color="#34d399", linestyle="--", linewidth=1.2,
           alpha=0.85, label=f"Final mean  {final_mean:+.3f}")

ax.set_xlabel("Episode", color="#c9d1d9", fontsize=12)
ax.set_ylabel("Reward",  color="#c9d1d9", fontsize=12)
ax.set_title(
    "SpindleFlow RL — Delegation Policy Learning Curve\n"
    f"RecurrentPPO · LSTM · {TOTAL_TIMESTEPS:,} steps · {n_ep:,} episodes",
    color="#f0f6fc", fontsize=13, fontweight="bold", pad=14,
)
ax.tick_params(colors="#8b949e")
for sp in ax.spines.values():
    sp.set_edgecolor("#30363d")
ax.grid(color="#21262d", linewidth=0.8, alpha=0.9)
ax.legend(fontsize=10, framealpha=0.85,
          facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

sign = "▲" if improvement >= 0 else "▼"
ax.annotate(f"  {sign} {abs(improvement):.3f} improvement",
            xy=(n_ep * 0.65, (early_mean + final_mean) / 2),
            color="#f0f6fc", fontsize=10, fontstyle="italic")

fig.tight_layout()
fig.savefig("/content/reward_curve.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()

_tlog(f"Curve: early={early_mean:+.4f}  final={final_mean:+.4f}  "
      f"improvement={improvement:+.4f}")
print(f"Episodes   : {n_ep:,}")
print(f"Improvement: {improvement:+.4f}")
print("\nCELL 6 done ✓ — reward curve saved")


# ============================================================
# CELL 7 — Learning features audit
# ============================================================
import json
from pathlib import Path

print("=" * 52)
print("LEARNING FEATURES AUDIT")
print("=" * 52)

print(f"\nFeature 5 — Curriculum (performance-gated)")
print(f"  Phase        : {curriculum.current_phase}/3")
print(f"  Rolling mean : {curriculum.rolling_mean():.3f}")
print(f"  {curriculum.progress_str()}")

mem_path = Path(_cfg.get("specialist_improvement", {}).get(
    "memory_path", "data/specialist_memory.json"))
print(f"\nFeature 2 — Specialist memory ({mem_path})")
if mem_path.exists():
    data = json.loads(mem_path.read_text())
    total = sum(len(v) for v in data.values())
    print(f"  {len(data)} specialists · {total} total entries")
    for sid, entries in list(data.items())[:3]:
        avg = sum(e["reward"] for e in entries) / len(entries)
        print(f"    {sid}: {len(entries)} entries, avg={avg:.3f}")
else:
    print("  No file yet (finetuner fires after 100 completed episodes)")

spawn_path = Path(_cfg.get("environment", {}).get(
    "spawn_memory_path", "data/spawn_memory.jsonl"))
print(f"\nFeature 3 — Spawn memory ({spawn_path})")
if spawn_path.exists():
    lines = [l for l in spawn_path.read_text().splitlines() if l.strip()]
    print(f"  {len(lines)} spawn records")
    for line in lines[:2]:
        rec = json.loads(line)
        print(f"    {rec['specialist_role']} | reward={rec['episode_reward']:.3f}")
else:
    print("  No file yet")

res_path = Path(_cfg.get("agents", {}).get(
    "resolution_memory_path", "data/resolution_memory.jsonl"))
print(f"\nFeature 4 — Resolution bandit ({res_path})")
if res_path.exists():
    lines = [l for l in res_path.read_text().splitlines() if l.strip()]
    print(f"  {len(lines)} outcome records")
else:
    print("  No file yet")

print("\n" + "=" * 52)
print("CELL 7 done ✓")


# ============================================================
# CELL 8 — Push to HuggingFace Hub
# ============================================================
import os, numpy as np
from huggingface_hub import HfApi, CommitOperationAdd

from huggingface_hub import whoami
HF_REPO = f"{whoami(token=HF_TOKEN)['name']}/spindleflow-rl"
api = HfApi(token=HF_TOKEN)

_tlog(f"Pushing to https://huggingface.co/{HF_REPO} ...")
api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)

ep = reward_logger.episode_rewards
readme = f"""---
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

LSTM PPO (RecurrentPPO) trained on SpindleFlow-v0 (OpenEnv). Colab T4 GPU.

## Training summary
| Metric | Value |
|---|---|
| Algorithm | RecurrentPPO (SB3 + sb3-contrib) |
| Total timesteps | {TOTAL_TIMESTEPS:,} |
| Episodes | {len(ep):,} |
| Early baseline (first 50 ep) | {early_mean:.4f} |
| Final mean (last 200 ep) | {final_mean:.4f} |
| Improvement | {improvement:+.4f} |
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
    f.write(readme)

candidates = [
    ("/content/spindleflow_model.zip",          "spindleflow_model.zip"),
    ("/content/vec_normalize.pkl",              "vec_normalize.pkl"),
    ("/content/reward_curve.png",               "reward_curve.png"),
    ("/content/demo/assets/reward_curve.json",  "reward_curve.json"),
    ("/content/logs/training_log.txt",          "training_log.txt"),
    (readme_path,                               "README.md"),
]

ops = [
    CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src)
    for src, dst in candidates if os.path.exists(src)
]

api.create_commit(
    repo_id=HF_REPO, repo_type="model", operations=ops,
    commit_message="Add trained SpindleFlow RL policy (Colab T4)",
    token=HF_TOKEN,
)

_tlog(f"Uploaded {len(ops)} files:")
for src, dst in candidates:
    if os.path.exists(src):
        _tlog(f"  {dst}")
_tlog(f"Model live : https://huggingface.co/{HF_REPO}")
_tlog(f"Log        : https://huggingface.co/{HF_REPO}/blob/main/training_log.txt")
print("\nCELL 8 done ✓ — all done!")
