"""
Hierarchical Factored Action Space.

4 heads decoded sequentially at each step:
  Head 1: Meta-action — what high-level thing to do?
  Head 2: Specialist selection — which specialist(s) to call?
  Head 3: Delegation mode — how to call them?
  Head 4: Mode parameters — how many rounds, threshold, etc.?

Design: Sequential decomposition keeps each head's distribution
tractable for PPO. The policy sees a flattened joint action, but
training uses the factored structure.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np


class MetaAction(IntEnum):
    """Top-level orchestrator decisions."""
    CALL_SPECIALIST  = 0    # Call one or more specialists
    STOP             = 1    # Stop delegation, synthesize output
    CALL_MEDIATOR    = 2    # Call conflict mediator
    CLARIFY_TASK     = 3    # Request task clarification (if ambiguous)
    DELEGATE_SUBTASK = 4    # Delegate a sub-problem (2nd level)
    RETRY_FAILED     = 5    # Retry a failed specialist with fallback
    PARALLEL_SPAWN   = 6    # Spawn parallel specialists
    SPAWN_SPECIALIST = 7    # Policy requests a new specialist be created


class DelegationMode(IntEnum):
    """How to execute the selected specialists."""
    SEQUENTIAL     = 0      # A → B → C (each sees previous output)
    PARALLEL       = 1      # A, B, C all run simultaneously
    FAN_OUT_REDUCE = 2      # A, B, C run → mediator reduces output
    ITERATIVE      = 3      # Run specialist, check output, loop until threshold
    CONDITIONAL    = 4      # Run A; if condition met, run B, else C
    PRIORITY_QUEUE = 5      # Run in priority order, stop when threshold met
    BROADCAST      = 6      # Send to all specialists, take first to complete


@dataclass
class FactoredAction:
    """
    The complete action decoded from all 4 heads.
    This is what gets passed to the environment's step() function.
    """
    meta_action: MetaAction
    specialist_ids: list[str]               # Which specialists to call
    delegation_mode: DelegationMode         # How to call them
    mode_params: dict                       # Mode-specific parameters
    raw_action: Optional[np.ndarray] = None # Raw policy output (for logging)

    def is_terminal(self) -> bool:
        """Returns True if this action ends the episode."""
        return self.meta_action == MetaAction.STOP

    def to_log_dict(self) -> dict:
        return {
            "meta_action": self.meta_action.name,
            "specialists": self.specialist_ids,
            "mode": self.delegation_mode.name,
            "params": self.mode_params,
        }


class ActionDecoder:
    """
    Decodes a flat action vector from the policy into a FactoredAction.

    Action vector layout:
      [0]                     : meta_action index (int, 0–6)
      [1 : 1+max_specialists] : specialist selection (multi-hot float)
      [1+max_specialists]     : delegation_mode index (int, 0–6)
      [2+max_specialists : *] : mode_params (continuous, 4 floats)

    Total action dim = 1 + max_specialists + 1 + 4 = max_specialists + 6
    """

    NUM_META_ACTIONS    = len(MetaAction)
    NUM_DELEGATION_MODES = len(DelegationMode)
    NUM_MODE_PARAMS     = 4

    def __init__(self, specialist_ids: list[str], max_specialists: int = 8):
        self.specialist_ids = specialist_ids
        self.max_specialists = min(len(specialist_ids), max_specialists)
        self.action_dim = self.max_specialists + 6

    def decode(
        self,
        action_vector: np.ndarray,
        valid_specialist_mask: Optional[np.ndarray] = None,
    ) -> FactoredAction:
        """
        Decode a flat action vector into a FactoredAction.

        Args:
            action_vector: Flat numpy array from the policy
            valid_specialist_mask: Binary mask, 1 = valid, 0 = masked out
                                   (enforces DAG constraints)
        """
        action_vector = np.asarray(action_vector, dtype=np.float32)

        # Head 1: Meta-action
        meta_idx = int(np.clip(round(action_vector[0]), 0, self.NUM_META_ACTIONS - 1))
        meta_action = MetaAction(meta_idx)

        # Head 2: Specialist selection (multi-hot)
        spec_logits = action_vector[1: 1 + self.max_specialists]
        if valid_specialist_mask is not None:
            spec_logits = spec_logits * valid_specialist_mask[:self.max_specialists]

        selected_indices = np.where(spec_logits > 0.0)[0]
        if len(selected_indices) == 0 and meta_action == MetaAction.CALL_SPECIALIST:
            # Fallback: select the highest-scoring specialist
            selected_indices = [int(np.argmax(spec_logits))]

        selected_ids = [
            self.specialist_ids[i]
            for i in selected_indices
            if i < len(self.specialist_ids)
        ]

        # Head 3: Delegation mode
        mode_idx = int(np.clip(
            round(action_vector[1 + self.max_specialists]),
            0, self.NUM_DELEGATION_MODES - 1
        ))
        delegation_mode = DelegationMode(mode_idx)

        # Head 4: Mode parameters
        param_start = 2 + self.max_specialists
        raw_params = action_vector[param_start: param_start + self.NUM_MODE_PARAMS]
        mode_params = self._decode_mode_params(delegation_mode, raw_params)

        return FactoredAction(
            meta_action=meta_action,
            specialist_ids=selected_ids,
            delegation_mode=delegation_mode,
            mode_params=mode_params,
            raw_action=action_vector,
        )

    def _decode_mode_params(
        self, mode: DelegationMode, raw_params: np.ndarray
    ) -> dict:
        """Decode mode-specific parameters from the raw continuous params."""
        p = np.clip(raw_params, 0.0, 1.0)
        if mode == DelegationMode.ITERATIVE:
            return {
                "max_rounds": int(1 + round(p[0] * 4)),          # 1–5 rounds
                "quality_threshold": float(0.5 + p[1] * 0.5),    # 0.5–1.0
            }
        elif mode == DelegationMode.PRIORITY_QUEUE:
            return {
                "stop_threshold": float(0.6 + p[0] * 0.4),       # 0.6–1.0
            }
        elif mode == DelegationMode.CONDITIONAL:
            return {
                "condition_threshold": float(0.4 + p[0] * 0.6),  # 0.4–1.0
            }
        else:
            return {"parallel_budget_ms": int(2000 + p[0] * 6000)}

    def get_action_dim(self) -> int:
        return self.action_dim

    def build_specialist_mask(
        self, valid_specialist_ids: list[str]
    ) -> np.ndarray:
        """Build a binary mask for valid specialist selections."""
        mask = np.zeros(self.max_specialists, dtype=np.float32)
        valid_set = set(valid_specialist_ids)
        for i, sid in enumerate(self.specialist_ids[: self.max_specialists]):
            if sid in valid_set:
                mask[i] = 1.0
        return mask
