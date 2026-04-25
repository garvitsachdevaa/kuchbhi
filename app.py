"""
SpindleFlow RL — HuggingFace Spaces Training App
=================================================
Upload this file + requirements.txt to a NEW HF Space.

Space settings:
  SDK       : Gradio
  Hardware  : A100 (large)  ← select when creating the Space
  Secrets   : HF_TOKEN        (write token — huggingface.co → Settings → Tokens)
              OPENAI_API_KEY  (optional — enables finetuner + spawn self-learning)
              HF_MODEL_REPO   (optional — defaults to <your-username>/spindleflow-rl)

Training starts automatically when the Space boots.
Refresh the page or click "Refresh" to see live progress.
"""

import gradio as gr
import threading
import os, sys, json, time
import numpy as np

# ── Shared state ─────────────────────────────────────────────
_logs   = []
_status = {"phase": "starting", "done": False, "error": None}
_LOG_FILE = "/home/user/app/assets/training_log.txt"


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _logs.append(line)
    print(line, flush=True)
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── Training thread ───────────────────────────────────────────
def _training_thread():
    try:
        # ── Tokens ──────────────────────────────────────────
        HF_TOKEN   = os.environ.get("HF_TOKEN", "")
        OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
        HF_REPO    = os.environ.get("HF_MODEL_REPO", "")

        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN secret not set. "
                "Go to Space Settings → Variables and secrets → add HF_TOKEN."
            )

        if OPENAI_KEY:
            _log("OpenAI key found — finetuner + spawn self-learning enabled.")
        else:
            _log("No OPENAI_API_KEY — running in simulation mode (fast training).")

        if not HF_REPO:
            from huggingface_hub import whoami
            username = whoami(token=HF_TOKEN)["name"]
            HF_REPO = f"{username}/spindleflow-rl"
        _log(f"Model will be pushed to: https://huggingface.co/{HF_REPO}")

        REPO_DIR = "/home/user/app"
        os.chdir(REPO_DIR)
        sys.path.insert(0, REPO_DIR)
        _log(f"Working directory: {REPO_DIR}")

        os.makedirs("/home/user/app/data",        exist_ok=True)
        os.makedirs("/home/user/app/checkpoints", exist_ok=True)
        os.makedirs("/home/user/app/assets",      exist_ok=True)

        # ── Create HF repo early so periodic pushes can start ──
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi()
        api.create_repo(repo_id=HF_REPO, repo_type="model",
                        exist_ok=True, token=HF_TOKEN)

        # ── Patch env for simulate_specialists ──────────────
        _log("Loading environment...")
        from env.spindleflow_env import SpindleFlowEnv
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

        # ── Smoke test ──────────────────────────────────────
        _log("Running smoke test...")
        env = SpindleFlowEnv(
            config_path="configs/training_config.yaml",
            catalog_path="configs/specialist_catalog.yaml",
            use_real_spindleflow=False,
            phase=1,
            simulate_specialists=True,
        )
        obs, info = env.reset()
        env.step(env.action_space.sample())
        env.close()
        _log(f"Smoke test OK — obs shape {obs.shape}")

        # ── Training ────────────────────────────────────────
        import torch, yaml
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
        from policy.lstm_policy import build_policy_kwargs
        from training.curriculum import CurriculumManager
        from training.specialist_improvement_callback import SpecialistImprovementCallback

        with open("configs/training_config.yaml") as f:
            cfg = yaml.safe_load(f)

        curriculum = CurriculumManager(config_path="configs/training_config.yaml")

        class RewardLogger(BaseCallback):
            def __init__(self, curriculum):
                super().__init__()
                self.episode_rewards = []
                self._running = 0.0
                self._curriculum = curriculum

            def _on_step(self):
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
                        if advanced or n % 25 == 0:
                            _log(
                                f"Ep {n:5d} | reward {ep:+.3f} | "
                                f"{self._curriculum.progress_str()}"
                            )
                return True

        class PeriodicHubPush(BaseCallback):
            """Pushes a checkpoint + log file to HF Hub every N steps.
            Ensures no work is lost if the Space is interrupted."""

            def __init__(self, api, hf_repo, hf_token, vec_env, push_every=50_000):
                super().__init__()
                self._api        = api
                self._repo       = hf_repo
                self._token      = hf_token
                self._vec_env    = vec_env
                self._push_every = push_every
                self._last_push  = 0

            def _on_step(self):
                if self.num_timesteps - self._last_push < self._push_every:
                    return True
                self._last_push = self.num_timesteps
                try:
                    _log(f"Periodic save at step {self.num_timesteps:,} ...")
                    self.model.save("/home/user/app/spindleflow_model_latest")
                    self._vec_env.save("/home/user/app/vec_normalize_latest.pkl")
                    candidates = [
                        ("/home/user/app/spindleflow_model_latest.zip", "spindleflow_model_latest.zip"),
                        ("/home/user/app/vec_normalize_latest.pkl",     "vec_normalize_latest.pkl"),
                        ("/home/user/app/assets/training_log.txt",      "training_log.txt"),
                    ]
                    ops = [
                        CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src)
                        for src, dst in candidates if os.path.exists(src)
                    ]
                    if ops:
                        self._api.create_commit(
                            repo_id=self._repo, repo_type="model",
                            operations=ops,
                            commit_message=f"Checkpoint at step {self.num_timesteps:,}",
                            token=self._token,
                        )
                        _log(f"Periodic push done — {len(ops)} files at step {self.num_timesteps:,}")
                except Exception as e:
                    _log(f"Periodic push failed (non-fatal): {e}")
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

        _ppo  = cfg.get("ppo",  {})
        _lstm = cfg.get("lstm", {})

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
            seed=int(cfg.get("training", {}).get("seed", 42)),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        _log(f"Training on : {model.device}")
        _log(f"Curriculum  : Phase {curriculum.current_phase} — {curriculum.progress_str()}")
        total_steps = int(cfg.get("training", {}).get("total_timesteps", 500_000))
        _log(f"Total steps : {total_steps:,}")
        _log("Training started...\n")
        _status["phase"] = "training"

        reward_logger  = RewardLogger(curriculum=curriculum)
        checkpoint_cb  = CheckpointCallback(
            save_freq=10_000, save_path="/home/user/app/checkpoints/"
        )
        improvement_cb = SpecialistImprovementCallback(
            improve_every_n_episodes=cfg.get("specialist_improvement", {}).get(
                "improve_every_n_episodes", 100
            ),
            verbose=1,
        )
        periodic_push  = PeriodicHubPush(
            api=api, hf_repo=HF_REPO, hf_token=HF_TOKEN,
            vec_env=vec_env, push_every=50_000,
        )

        model.learn(
            total_timesteps=total_steps,
            callback=[reward_logger, checkpoint_cb, improvement_cb, periodic_push],
        )

        MODEL_PATH = "/home/user/app/spindleflow_model"
        STATS_PATH = "/home/user/app/vec_normalize.pkl"
        model.save(MODEL_PATH)
        vec_env.save(STATS_PATH)
        _log(f"Model saved — {len(reward_logger.episode_rewards)} episodes completed.")
        _log(f"Final curriculum: {curriculum.progress_str()}")

        # ── Reward curve ────────────────────────────────────
        _status["phase"] = "saving"
        ep_rewards = reward_logger.episode_rewards or [0.0]
        episodes   = list(range(len(ep_rewards)))
        window     = max(50, len(ep_rewards) // 20)
        smoothed   = [
            float(np.mean(ep_rewards[max(0, i - window):i + 1]))
            for i in range(len(ep_rewards))
        ]

        step = max(1, len(episodes) // 200)
        with open("/home/user/app/assets/reward_curve.json", "w") as f:
            json.dump({
                "episodes":     episodes[::step],
                "mean_rewards": smoothed[::step],
            }, f)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plot_every = max(1, len(ep_rewards) // 500)
        plt.plot(episodes[::plot_every], ep_rewards[::plot_every],
                 "o", markersize=2, alpha=0.2, color="#00d4ff", label="Episode reward")
        plt.plot(episodes[::plot_every], smoothed[::plot_every],
                 linewidth=2.5, color="#ff6b35", label=f"Smoothed ({window}-ep mean)")
        plt.axhline(y=float(np.mean(ep_rewards[:5])),
                    color="#94a3b8", linestyle="--", alpha=0.8, label="Early baseline")
        plt.axhline(y=float(np.mean(ep_rewards[-200:])),
                    color="#34d399", linestyle="--", alpha=0.8, label="Final mean")
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.title("SpindleFlow RL — Delegation Policy Learning Curve")
        plt.legend(); plt.grid(alpha=0.2); plt.tight_layout()
        plt.savefig("/home/user/app/assets/reward_curve.png", dpi=150)
        plt.close()
        _log("Reward curve saved.")

        # ── Push everything to HF Hub ────────────────────────
        _status["phase"] = "uploading"
        _log(f"Pushing to https://huggingface.co/{HF_REPO} ...")

        ep  = reward_logger.episode_rewards
        f5  = float(np.mean(ep[:5]))  if len(ep) >= 5 else 0.0
        l5  = float(np.mean(ep[-5:])) if len(ep) >= 5 else 0.0
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

LSTM PPO agent trained on SpindleFlow-v0 (OpenEnv).

## Training summary
| Metric | Value |
|---|---|
| Algorithm | RecurrentPPO (SB3 + sb3-contrib) |
| Total timesteps | {total_steps:,} |
| Episodes completed | {len(ep)} |
| First-5 mean reward | {f5:.4f} |
| Last-5 mean reward | {l5:.4f} |
| Improvement | {l5 - f5:+.4f} |
| Device | {str(model.device)} |

![Reward Curve](reward_curve.png)

## Load
```python
from sb3_contrib import RecurrentPPO
from huggingface_hub import hf_hub_download
model = RecurrentPPO.load(hf_hub_download("{HF_REPO}", "spindleflow_model.zip"))
```
"""
        with open("/home/user/app/README.md", "w") as f:
            f.write(readme)

        candidates = [
            ("/home/user/app/spindleflow_model.zip",        "spindleflow_model.zip"),
            ("/home/user/app/vec_normalize.pkl",            "vec_normalize.pkl"),
            ("/home/user/app/assets/reward_curve.png",      "reward_curve.png"),
            ("/home/user/app/assets/reward_curve.json",     "reward_curve.json"),
            ("/home/user/app/assets/training_log.txt",      "training_log.txt"),
            ("/home/user/app/README.md",                    "README.md"),
            ("/home/user/app/data/specialist_memory.json",  "data/specialist_memory.json"),
            ("/home/user/app/data/spawn_memory.jsonl",      "data/spawn_memory.jsonl"),
            ("/home/user/app/data/resolution_memory.jsonl", "data/resolution_memory.jsonl"),
        ]

        ops = [
            CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src)
            for src, dst in candidates
            if os.path.exists(src)
        ]
        api.create_commit(
            repo_id=HF_REPO, repo_type="model", operations=ops,
            commit_message="Add trained SpindleFlow RL policy",
            token=HF_TOKEN,
        )

        _log(f"Uploaded {len(ops)} files.")
        _log(f"Model live at: https://huggingface.co/{HF_REPO}")
        _status["done"] = True
        _status["phase"] = "complete"

    except Exception as exc:
        import traceback
        _log(f"ERROR: {exc}")
        _log(traceback.format_exc())
        _status["error"] = str(exc)
        _status["phase"] = "error"


# ── Start training immediately on Space boot ──────────────────
_thread = threading.Thread(target=_training_thread, daemon=True)
_thread.start()


# ── Gradio UI ─────────────────────────────────────────────────
def _get_state():
    phase = _status["phase"]
    if _status["done"]:
        label = "✅  Training complete — model pushed to HF Hub"
    elif _status["error"]:
        label = f"❌  Error: {_status['error']}"
    else:
        icons = {
            "starting": "⏳", "training": "🔄",
            "saving": "💾", "uploading": "📤",
        }
        label = f"{icons.get(phase, '🔄')}  {phase.capitalize()}..."
    return label, "\n".join(_logs[-120:])


with gr.Blocks(title="SpindleFlow RL Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SpindleFlow RL — Training Dashboard")
    gr.Markdown(
        "Training runs automatically on startup. "
        "Click **Refresh** every 30 s to see progress. "
        "When complete the model is pushed to your HF Hub repo."
    )

    with gr.Row():
        status_box = gr.Textbox(label="Status", value="⏳  Starting...",
                                interactive=False, scale=3)
        refresh_btn = gr.Button("🔄  Refresh", scale=1, variant="primary")

    log_box = gr.Textbox(
        label="Training log (last 120 lines)",
        value="",
        lines=30,
        max_lines=40,
        interactive=False,
    )

    refresh_btn.click(fn=_get_state, outputs=[status_box, log_box])
    demo.load(fn=_get_state, outputs=[status_box, log_box])

demo.launch()
