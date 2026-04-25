"""
Main training entry point.
Uses SB3 RecurrentPPO (LSTM) with curriculum learning.
"""

from __future__ import annotations
import os
import sys
import yaml
import click
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@click.command()
@click.option("--config", default="configs/training_config.yaml", help="Training config path")
@click.option("--phase", default=1, type=int, help="Starting curriculum phase (1/2/3)")
@click.option("--timesteps", default=None, type=int, help="Override total timesteps")
@click.option("--demo-mode", is_flag=True, help="Use real SpindleFlow (slower, for demo)")
@click.option("--checkpoint", default=None, help="Resume from checkpoint path")
def train(config, phase, timesteps, demo_mode, checkpoint):
    """Train the SpindleFlow RL delegation policy."""
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        print("ERROR: sb3-contrib required. Run: pip install sb3-contrib")
        sys.exit(1)

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, BaseCallback
    )
    from training.curriculum import CurriculumManager
    from training.specialist_improvement_callback import SpecialistImprovementCallback
    from env.spindleflow_env import SpindleFlowEnv
    from policy.lstm_policy import build_policy_kwargs

    with open(config) as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = cfg["ppo"]
    training_cfg = cfg["training"]
    lstm_cfg = cfg["lstm"]

    total_ts = timesteps or training_cfg["total_timesteps"]
    curriculum = CurriculumManager(config_path=config)

    print(f"\n{'='*60}")
    print(f"SpindleFlow RL Training")
    print(f"  Phase: {phase}")
    print(f"  Timesteps: {total_ts}")
    print(f"  Demo mode (real SpindleFlow): {demo_mode}")
    print(f"{'='*60}\n")

    def make_env():
        return SpindleFlowEnv(
            config_path=config,
            phase=phase,
            use_real_spindleflow=demo_mode,
        )

    n_envs = training_cfg.get("n_envs", 1)
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    policy_kwargs = build_policy_kwargs(
        hidden_size=lstm_cfg["hidden_size"],
        num_lstm_layers=lstm_cfg["num_layers"],
    )

    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        model = RecurrentPPO.load(checkpoint, env=env)
    else:
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tensorboard_logs/",
            verbose=1,
            seed=training_cfg["seed"],
            device=training_cfg["device"],
        )

    _max_specialists = cfg["environment"].get("max_specialists_per_episode", 6)

    class _RewardLogger(BaseCallback):
        def __init__(self, max_specialists: int, curriculum: CurriculumManager):
            super().__init__()
            self.episode_rewards: list[float] = []
            self.episode_entropies: list[float] = []
            self._running_reward = 0.0
            self._running_entropy: list[float] = []
            self._max_specialists = max_specialists
            self._curriculum = curriculum

        def _on_step(self):
            import numpy as np
            rewards = self.locals.get("rewards", [])
            dones   = self.locals.get("dones",   [])
            actions = self.locals.get("actions", None)
            if actions is not None:
                for action_vec in actions:
                    n = self._max_specialists
                    logits = action_vec[1:1 + n]
                    logits = logits - logits.max()
                    exp_l  = np.exp(logits)
                    probs  = exp_l / (exp_l.sum() + 1e-8)
                    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
                    self._running_entropy.append(entropy)
            for r, d in zip(rewards, dones):
                self._running_reward += float(r)
                if d:
                    ep_reward = self._running_reward
                    self.episode_rewards.append(ep_reward)
                    if self._running_entropy:
                        self.episode_entropies.append(
                            float(sum(self._running_entropy) / len(self._running_entropy))
                        )
                        self._running_entropy = []
                    self._running_reward = 0.0
                    self._curriculum.on_episode_end(ep_reward)
            return True

    reward_logger = _RewardLogger(max_specialists=_max_specialists, curriculum=curriculum)
    checkpoint_cb = CheckpointCallback(
        save_freq=2000,
        save_path="./checkpoints/",
        name_prefix="spindleflow_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best/",
        log_path="./eval_logs/",
        eval_freq=1000,
        n_eval_episodes=5,
        verbose=1,
    )
    si_cfg = cfg.get("specialist_improvement", {})
    improvement_cb = SpecialistImprovementCallback(
        improve_every_n_episodes=si_cfg.get("improve_every_n_episodes", 100),
        verbose=1,
    )

    print(f"Starting training for {total_ts} timesteps...")
    print(f"TensorBoard: tensorboard --logdir tensorboard_logs/\n")

    model.learn(
        total_timesteps=total_ts,
        callback=[checkpoint_cb, eval_cb, reward_logger, improvement_cb],
        reset_num_timesteps=checkpoint is None,
    )

    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/spindleflow_final")
    env.save("checkpoints/vec_normalize.pkl")
    print("\nTraining complete. Model saved to checkpoints/spindleflow_final")

    # Save reward curve for the Streamlit dashboard
    import json, numpy as np
    ep = reward_logger.episode_rewards
    if ep:
        os.makedirs("demo/assets", exist_ok=True)
        step = max(1, len(ep) // 200)
        smoothed = [float(np.mean(ep[max(0, i-19):i+1])) for i in range(len(ep))]
        with open("demo/assets/reward_curve.json", "w") as f:
            json.dump({"episodes": list(range(len(ep)))[::step],
                       "mean_rewards": smoothed[::step]}, f)
        print(f"Saved demo/assets/reward_curve.json  ({len(ep)} episodes)")

    # Save entropy log for Training tab entropy chart
    ep_e = reward_logger.episode_entropies
    if ep_e:
        step_e = max(1, len(ep_e) // 200)
        with open("demo/assets/entropy_log.json", "w") as f:
            json.dump({
                "episodes":       list(range(len(ep_e)))[::step_e],
                "mean_entropies": ep_e[::step_e],
            }, f)
        print(f"Saved demo/assets/entropy_log.json  ({len(ep_e)} episodes)")


if __name__ == "__main__":
    train()
