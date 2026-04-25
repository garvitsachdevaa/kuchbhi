"""
LSTM PPO Policy — POMDP-safe policy for SpindleFlow delegation.

Why LSTM: The scratchpad creates partial observability. Without recurrent
memory, the policy can't distinguish between "I just called backend_api"
and "I called backend_api 3 steps ago." The LSTM hidden state carries
this temporal context safely.

Implementation: Uses Stable Baselines 3's RecurrentPPO (sb3-contrib).
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import numpy as np


def build_policy_kwargs(
    hidden_size: int = 256,
    num_lstm_layers: int = 1,
) -> dict:
    """
    Build policy_kwargs for SB3 RecurrentPPO.
    Uses LSTM policy network with custom encoder.
    """
    return {
        "lstm_hidden_size": hidden_size,
        "n_lstm_layers": num_lstm_layers,
        "shared_lstm": False,
        "enable_critic_lstm": True,
        "net_arch": {"pi": [256, 128], "vf": [256, 128]},
    }
