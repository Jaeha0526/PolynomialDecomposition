"""
Beam Grouped Relative Policy Optimization.

Public entry point is ``BGRPOTrainer``; CLI is ``Training/BGRPO/run_bgrpo.py``.

Design:
  * Single-GPU. Model is 5-36M params; no FSDP / Ray / DeepSpeed.
  * Beam rollout is the policy's sampling step (explicit ``beam_width``).
  * Reward shape, decay constant, and wrong-answer penalty are config.
  * Frozen reference policy + PPO-clipped objective with Schulman k1 KL.
"""

from .reward import RewardConfig, RewardFn, build_reward  # noqa: F401
from .objective import GRPOObjective, ObjectiveConfig  # noqa: F401
from .rollout import BeamRollout, RolloutResult  # noqa: F401
from .trainer import BGRPOConfig, BGRPOTrainer  # noqa: F401
