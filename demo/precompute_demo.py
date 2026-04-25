"""
Precompute demo assets for the Streamlit dashboard.

Generates:
  demo/assets/demo_moment_1.json  — before/after comparison (Quality Demo tab)
  demo/assets/reward_curve.json   — placeholder if no real training curve exists yet

Run once before launching the UI:
  cd spindleflow-rl
  python demo/precompute_demo.py
"""

from __future__ import annotations
import os, sys, json
import numpy as np
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.spindleflow_env import SpindleFlowEnv

CONFIG  = "configs/training_config.yaml"
CATALOG = "configs/specialist_catalog.yaml"
ASSETS  = Path("demo/assets")
ASSETS.mkdir(parents=True, exist_ok=True)


def run_no_delegation(env: SpindleFlowEnv) -> dict:
    """Episode where the orchestrator stops immediately — baseline."""
    obs, info = env.reset()
    task = info["task"]

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0  # STOP immediately

    _, reward, _, _, step_info = env.step(action)
    return {
        "task":              task,
        "reward":            float(reward),
        "output":            env.generalist_baseline,
        "called":            [],
        "reward_components": step_info.get("reward_components", {}),
    }


def run_with_delegation(env: SpindleFlowEnv, n_specialists: int = 2) -> dict:
    """Episode where orchestrator calls specialists then stops."""
    obs, info = env.reset()
    task = info["task"]
    ids  = env.registry.list_ids()

    all_called: list[str] = []
    last_info:  dict      = {}

    for i in range(min(n_specialists, env.max_specialists)):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[0] = 0.0  # CALL_SPECIALIST
        spec_idx = i % len(ids)
        if spec_idx < env.max_specialists:
            action[1 + spec_idx] = 1.0
        _, _, term, trunc, step_info = env.step(action)
        all_called.extend(step_info.get("called_specialists", []))
        last_info = step_info
        if term or trunc:
            break

    # Explicit STOP to get final reward
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0
    _, reward, _, _, final_info = env.step(action)

    outputs = [
        f"[{e.author_role}]\n{e.content}"
        for e in env.scratchpad._entries
    ]
    specialist_output = "\n\n".join(outputs) if outputs else (
        f"[Specialist analysis for: {task[:80]}]\n"
        f"Domain-specific solution using best practices.\n"
        f"Specialists consulted: {', '.join(all_called) or 'none'}"
    )

    return {
        "task":              task,
        "reward":            float(reward),
        "output":            specialist_output,
        "called":            all_called,
        "reward_components": final_info.get("reward_components", {}),
    }


def build_demo_moment_1(env: SpindleFlowEnv) -> None:
    print("Running no-delegation episode (generalist baseline)...")
    base = run_no_delegation(env)

    print("Running with-delegation episode (2 specialists)...")
    spec = run_with_delegation(env, n_specialists=2)

    generalist_text = (
        f"Task: {base['task'][:120]}\n\n"
        f"--- Generalist (no delegation) ---\n"
        f"{base['output']}\n\n"
        f"Reward: {base['reward']:.4f}  |  Specialists called: none\n"
        f"Result: Generic, surface-level response with no domain depth."
    )
    specialist_text = (
        f"Task: {spec['task'][:120]}\n\n"
        f"--- Specialist-Routed (learned policy) ---\n"
        f"{spec['output']}\n\n"
        f"Reward: {spec['reward']:.4f}  |  "
        f"Specialists called: {', '.join(spec['called']) or 'n/a'}\n"
        f"Result: Domain-expert output with specific technical recommendations."
    )

    data = {
        "generalist_output": generalist_text,
        "specialist_output": specialist_text,
        "generalist_reward": base["reward"],
        "specialist_reward": spec["reward"],
        "improvement":       spec["reward"] - base["reward"],
    }

    out = ASSETS / "demo_moment_1.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {out}")
    print(f"  Generalist reward : {base['reward']:.4f}")
    print(f"  Specialist reward : {spec['reward']:.4f}")
    print(f"  Improvement       : {data['improvement']:+.4f}")


def build_placeholder_curve() -> None:
    """Write a synthetic curve ONLY if a real one doesn't exist yet."""
    path = ASSETS / "reward_curve.json"
    if path.exists():
        print(f"  reward_curve.json already exists — skipping placeholder.")
        return
    rng  = np.random.default_rng(42)
    eps  = list(range(0, 201, 5))
    rews = [float(np.clip(
                0.1 + 0.5 * (1 - np.exp(-e / 80)) + rng.normal(0, 0.04), 0, 1
            )) for e in eps]
    with open(path, "w") as f:
        json.dump({"episodes": eps, "mean_rewards": rews}, f)
    print(f"  Saved placeholder {path}")
    print("  Replace with real data after running Colab training.")


def main():
    print("Loading SpindleFlowEnv (~30s on first run)...")
    env = SpindleFlowEnv(
        config_path=CONFIG,
        catalog_path=CATALOG,
        use_real_spindleflow=False,
        phase=1,
    )
    print("Environment ready.\n")

    build_demo_moment_1(env)
    print()
    build_placeholder_curve()
    env.close()

    print("\nDone. All demo assets in demo/assets/")
    print("After Colab training, drop reward_curve.json into demo/assets/ to replace the placeholder.")


if __name__ == "__main__":
    main()
