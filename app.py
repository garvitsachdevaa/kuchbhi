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

import sys, os, subprocess
print("=== PYTHON STARTED ===", flush=True)

# ── Force CUDA torch before any `import torch` happens in this process ─────────
# requirements.txt installs CPU torch as a transitive dep of sentence-transformers.
# --force-reinstall overrides "already satisfied" and also installs nvidia-cudnn-cu12
# and other CUDA runtime packages needed by the LSTM kernel (cuDNN).
# This subprocess runs before gradio (and therefore before any torch import).
print("Installing CUDA torch + cuDNN...", flush=True)
_cuda_r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "--force-reinstall",
     "--index-url", "https://download.pytorch.org/whl/cu121",
     "--extra-index-url", "https://pypi.org/simple",
     "torch"],
    capture_output=True, text=True,
    timeout=600,
)
if _cuda_r.returncode == 0:
    print("CUDA torch + cuDNN installed OK.", flush=True)
else:
    print("CUDA torch install FAILED:", _cuda_r.stderr[-400:], flush=True)

import gradio as gr
print("=== GRADIO IMPORTED ===", flush=True)

import threading
import json, time
import numpy as np

# ── Persistent state (file-based so Gradio worker processes can read it) ─────
_LOG_FILE    = "/home/user/app/assets/training_log.txt"
_STATUS_FILE = "/home/user/app/assets/training_status.json"


def _write_status(phase, done=False, error=None):
    try:
        os.makedirs("/home/user/app/assets", exist_ok=True)
        with open(_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump({"phase": phase, "done": done, "error": error}, f)
    except Exception:
        pass


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs("/home/user/app/assets", exist_ok=True)
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
            username = whoami(token=HF_TOKEN)["name"].strip()
            HF_REPO = f"{username}/spindleflow-rl"
        else:
            HF_REPO = HF_REPO.strip()
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
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)

        # ── Force SentenceTransformer onto CUDA ─────────────
        # encode() is called every step (scratchpad) + per specialist call.
        # On CPU this costs ~250 ms/call → ~1 s/step. On CUDA it's ~10 ms.
        _log("Patching SentenceTransformer to CUDA...")
        import torch as _torch_st
        if _torch_st.cuda.is_available():
            try:
                from sentence_transformers import SentenceTransformer as _ST
                _orig_st_init = _ST.__init__
                def _fast_st_init(self, *args, **kwargs):
                    kwargs.setdefault("device", "cuda")
                    _orig_st_init(self, *args, **kwargs)
                _ST.__init__ = _fast_st_init
                _log("SentenceTransformer → cuda ✓")
            except Exception as _ep:
                _log(f"ST patch skipped: {_ep}")
        else:
            _log("WARNING: CUDA not available for SentenceTransformer — CPU mode (slow)")

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

        # ── Disable Tier-2 LLM scoring during training ───────
        # TieredRewardScorer._tier2_score calls OpenAI API (>1000ms per episode).
        # Returning None forces it to fall back to Tier-1 embedding scoring (~fast),
        # preserving a meaningful reward signal without API latency.
        from reward.tiered_reward import TieredRewardScorer
        TieredRewardScorer._get_openai_client = lambda self: None
        _log("TieredRewardScorer → Tier-1 only (LLM judge disabled for speed) ✓")

        # ── Patch generalist baseline → static (0 API calls per episode) ─────
        from env.spindleflow_env import SpindleFlowEnv as _SFEnv
        _STATIC_BASELINE = (
            "General problem-solving approach:\n"
            "1. Gather and clarify requirements\n"
            "2. Research common solution patterns\n"
            "3. Draft a high-level architecture\n"
            "4. Implement in small, testable increments\n"
            "5. Validate against acceptance criteria and deploy\n"
            "No specialist domain expertise applied."
        )
        _SFEnv._generate_generalist_baseline = lambda self, task: _STATIC_BASELINE
        _log("Generalist baseline → static simulation (0 API calls per episode) ✓")

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
        _log(f"Smoke test OK — obs shape {obs.shape}")

        # ── Benchmark: encode speed and full step speed ──────
        _log("Benchmarking SentenceTransformer encode speed...")
        _N_enc = 50
        _t0 = time.perf_counter()
        for _ in range(_N_enc):
            env.registry.embed_query("Software engineering task requiring specialist delegation")
        _enc_ms = (time.perf_counter() - _t0) / _N_enc * 1000
        _enc_device = "CUDA ✓ fast" if _enc_ms < 50 else "CPU — slow, patch may have failed"
        _log(f"Encode speed  : {_enc_ms:.1f} ms/call  [{_enc_device}]")

        _log("Benchmarking full env.step() speed...")
        _N_steps = 30
        obs_b, _ = env.reset()
        _t0 = time.perf_counter()
        for _ in range(_N_steps):
            obs_b, _, _d, _, _ = env.step(env.action_space.sample())
            if _d:
                obs_b, _ = env.reset()
        _step_ms = (time.perf_counter() - _t0) / _N_steps * 1000
        _step_ok = "fast ✓" if _step_ms < 100 else "slow — check logs"
        _log(f"Step speed    : {_step_ms:.1f} ms/step [{_step_ok}]")
        _log(f"Projected 100k steps: {100_000 * _step_ms / 1000 / 60:.0f} min")
        env.close()

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
            """Pushes checkpoint + log + reward curve to HF Hub every N steps."""

            def __init__(self, api, hf_repo, hf_token, vec_env,
                         reward_logger_ref, push_every=10_000):
                super().__init__()
                self._api           = api
                self._repo          = hf_repo
                self._token         = hf_token
                self._vec_env       = vec_env
                self._rl_ref        = reward_logger_ref
                self._push_every    = push_every
                self._last_push     = 0

            def _save_curve(self):
                ep = self._rl_ref.episode_rewards
                if len(ep) < 2:
                    return
                window  = max(10, len(ep) // 20)
                smoothed = [
                    float(np.mean(ep[max(0, i - window):i + 1]))
                    for i in range(len(ep))
                ]
                step = max(1, len(ep) // 200)
                with open("/home/user/app/assets/reward_curve.json", "w") as f:
                    json.dump({
                        "episodes":     list(range(len(ep)))[::step],
                        "mean_rewards": smoothed[::step],
                        "raw_rewards":  ep[::step],
                        "step":         self.num_timesteps,
                    }, f)
                import matplotlib, matplotlib.pyplot as plt
                matplotlib.use("Agg")
                plt.figure(figsize=(10, 4))
                every = max(1, len(ep) // 500)
                plt.plot(range(0, len(ep), every), ep[::every],
                         "o", markersize=2, alpha=0.2, color="#00d4ff",
                         label="Episode reward")
                plt.plot(range(0, len(ep), every), smoothed[::every],
                         linewidth=2.5, color="#ff6b35",
                         label=f"Smoothed ({window}-ep mean)")
                if len(ep) >= 5:
                    plt.axhline(float(np.mean(ep[:5])),
                                color="#94a3b8", linestyle="--", alpha=0.8,
                                label="Early baseline")
                    plt.axhline(float(np.mean(ep[-min(200, len(ep)):])),
                                color="#34d399", linestyle="--", alpha=0.8,
                                label="Current mean")
                plt.xlabel("Episode"); plt.ylabel("Reward")
                plt.title(f"SpindleFlow RL — Learning Curve (step {self.num_timesteps:,})")
                plt.legend(); plt.grid(alpha=0.2); plt.tight_layout()
                plt.savefig("/home/user/app/assets/reward_curve.png", dpi=150)
                plt.close()

            def _on_step(self):
                if self.num_timesteps - self._last_push < self._push_every:
                    return True
                self._last_push = self.num_timesteps
                try:
                    _log(f"Periodic save at step {self.num_timesteps:,} ...")
                    self.model.save("/home/user/app/spindleflow_model_latest")
                    self._vec_env.save("/home/user/app/vec_normalize_latest.pkl")
                    self._save_curve()
                    candidates = [
                        ("/home/user/app/spindleflow_model_latest.zip", "spindleflow_model_latest.zip"),
                        ("/home/user/app/vec_normalize_latest.pkl",     "vec_normalize_latest.pkl"),
                        ("/home/user/app/assets/training_log.txt",      "training_log.txt"),
                        ("/home/user/app/assets/reward_curve.json",     "reward_curve.json"),
                        ("/home/user/app/assets/reward_curve.png",      "reward_curve.png"),
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

        _n_envs = int(cfg.get("training", {}).get("n_envs", 1))
        vec_env = DummyVecEnv([make_env] * _n_envs)
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
        total_steps = 30_000  # ~45 min on A100 with simulation, produces clean reward curve
        _log(f"Total steps : {total_steps:,}")
        _log("Training started...\n")
        _write_status("training")

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
            vec_env=vec_env, reward_logger_ref=reward_logger, push_every=5_000,
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
        _write_status("saving")
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
        _write_status("uploading")
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
        _write_status("complete", done=True)

    except Exception as exc:
        import traceback
        _log(f"ERROR: {exc}")
        _log(traceback.format_exc())
        _write_status("error", error=str(exc))


# ── Start training immediately on Space boot ──────────────────
_thread = threading.Thread(target=_training_thread, daemon=True)
_thread.start()


# ── Gradio UI ─────────────────────────────────────────────────
def _get_state():
    # Read from files — works across Gradio worker processes
    try:
        with open(_STATUS_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
        phase, done, error = s.get("phase", "starting"), s.get("done", False), s.get("error")
    except Exception:
        phase, done, error = "starting", False, None

    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        log_text = "".join(lines[-120:])
    except Exception:
        log_text = ""

    if done:
        label = "✅  Training complete — model pushed to HF Hub"
    elif error:
        label = f"❌  Error: {error}"
    else:
        icons = {"starting": "⏳", "training": "🔄", "saving": "💾", "uploading": "📤"}
        label = f"{icons.get(phase, '🔄')}  {phase.capitalize()}..."
    return label, log_text


CSS = """
body, .gradio-container { background: #0f172a !important; }
.status-box textarea {
    font-size: 1.05rem !important; font-weight: 700 !important;
    background: #1e293b !important; color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
}
.log-box textarea {
    font-family: 'Courier New', monospace !important;
    font-size: 0.8rem !important; line-height: 1.5 !important;
    background: #0f172a !important; color: #94a3b8 !important;
    border: 1px solid #1e293b !important;
}
h1 { color: #f1f5f9 !important; }
p, label { color: #94a3b8 !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="SpindleFlow RL Training", css=CSS) as demo:
    gr.Markdown("# 🤖 SpindleFlow RL — Training Dashboard")
    gr.Markdown(
        "Live training log — updates every 10 s automatically. "
        "When complete the trained model is pushed to your HF Hub repo."
    )

    with gr.Row():
        status_box = gr.Textbox(
            label="Status",
            value="⏳  Starting...",
            interactive=False,
            scale=4,
            elem_classes="status-box",
        )
        refresh_btn = gr.Button("🔄  Refresh now", scale=1, variant="primary")

    log_box = gr.Textbox(
        label="Training log (last 120 lines)",
        value="",
        lines=28,
        max_lines=40,
        interactive=False,
        elem_classes="log-box",
    )

    refresh_btn.click(fn=_get_state, outputs=[status_box, log_box])
    demo.load(fn=_get_state, outputs=[status_box, log_box])
    timer = gr.Timer(value=10)
    timer.tick(fn=_get_state, outputs=[status_box, log_box])

demo.launch()
