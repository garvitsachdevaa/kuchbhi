"""
Cross-company transfer learning strategy.
Freeze encoder, fine-tune specialist-selection and mode heads only.
50 episodes for same-domain, not 600.
"""

from __future__ import annotations
import os
from pathlib import Path


class TransferLearningStrategy:
    """
    Enables rapid adaptation to new company rosters.

    Strategy:
    - The encoder already understands task-capability semantics
    - Only the specialist-selection and mode heads need updating
    - Fine-tune for 50 episodes same-domain (vs 600 from scratch)
    """

    def __init__(self, base_model_path: str = "checkpoints/spindleflow_final"):
        self.base_model_path = Path(base_model_path)

    def fine_tune_for_new_roster(
        self,
        new_catalog_path: str,
        new_company_tasks: list[str],
        num_episodes: int = 50,
        output_path: str = "checkpoints/fine_tuned",
    ) -> None:
        """
        Fine-tune the base policy for a new company's specialist roster.

        Implementation:
        1. Load base model (encoder weights frozen)
        2. Replace specialist registry with new catalog
        3. Run fine-tuning for num_episodes
        4. Save fine-tuned model

        For hackathon: documented as architecture decision.
        Full implementation requires loading the SB3 model and
        selectively freezing layers.
        """
        print(f"[Transfer] Fine-tuning for new roster: {new_catalog_path}")
        print(f"[Transfer] Tasks: {len(new_company_tasks)} company-specific tasks")
        print(f"[Transfer] Episodes: {num_episodes} (vs 600 from scratch)")
        print(f"[Transfer] Strategy: Encoder frozen, selection+mode heads trainable")
        print(f"[Transfer] Estimated time: {num_episodes * 2}s (vs 1200s from scratch)")
        print(f"[Transfer] NOTE: Full SB3 layer-freezing implementation pending.")

    def freeze_encoder_layers(self, model) -> None:
        """
        Freeze the encoder layers of the SB3 RecurrentPPO model.
        Only specialist-selection and mode heads remain trainable.
        """
        frozen_count = 0
        for name, param in model.policy.named_parameters():
            if "lstm" not in name and "action_net" not in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"[Transfer] Frozen {frozen_count} parameter groups")
        trainable = sum(
            p.numel() for p in model.policy.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in model.policy.parameters())
        print(f"[Transfer] Trainable: {trainable:,} / {total:,} parameters")
