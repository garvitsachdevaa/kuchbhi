# SpindleFlow RL — Google Colab Quick Start

## How to run the training notebook

1. Open [Google Colab](https://colab.research.google.com/)
2. Runtime > Change runtime type > **T4 GPU** (free tier)
3. Clone this repo into Colab:
   ```python
   !git clone https://github.com/YOUR_USERNAME/spindleflow-rl.git
   %cd spindleflow-rl
   ```
4. Run cells 1–6 in `colab/train_colab.py` sequentially
5. Cell 6 produces `reward_curve.png` — download it for your HuggingFace blog post

## What the Colab script demonstrates

- OpenEnv environment registration and compliance check
- HuggingFace TRL PPOConfig initialization
- SB3 RecurrentPPO training (5,000-step demo, scalable to 100,000)
- Reward improvement curve (observable evidence for judging criterion 3)

## Full training run

Change `total_timesteps=5_000` to `total_timesteps=100_000` for the full run.
Use a Colab Pro instance or a local GPU for the full 100k-step run.

## Before you submit

Replace `YOUR_USERNAME` in the clone URL with your actual GitHub username,
then share the Colab link in your HuggingFace blog post.
