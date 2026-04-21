"""
Reward functions for BGRPO.

Three shapes, all pluggable via :class:`RewardConfig`:

  * ``simple``        — r = correct_reward if correct else wrong_penalty.
  * ``rank``          — r = correct_reward * exp(-rank / decay_base) if correct.
  * ``reverse_rank``  — negative control: r = correct_reward * exp(+rank / decay_base);
    top beams are *penalized* when correct. Used to verify the rank signal
    is load-bearing and not just noise.

The reward function is a pure map ``correctness_mask -> reward_vector`` —
the trainer owns sympy/Mathematica-style validation upstream.

Usage::

    reward_fn = build_reward(RewardConfig(kind="rank", beam_width=32), device=device)
    rewards = reward_fn(correct_mask)   # shape (beam_width,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


RewardFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class RewardConfig:
    kind: str = "simple"            # "simple" | "rank" | "reverse_rank"
    beam_width: int = 32
    decay_base: int | None = None   # defaults to beam_width if None
    correct_reward: float = 1.0     # reward magnitude before rank decay
    wrong_penalty: float = 0.0      # reward for incorrect beams (paper: 0.0)
    adjust_rewards: bool = False    # if True, rescale so sum(r) == 0 per group


def build_reward(cfg: RewardConfig, device: Optional[torch.device] = None) -> RewardFn:
    """Return a reward function ``correct_mask -> reward_vector``.

    ``correct_mask`` is a bool/0-1 tensor of shape (beam_width,); output is
    a float tensor of the same shape. ``device`` pre-places the internal
    per-rank lookup so each reward call avoids a H2D transfer.
    """
    if cfg.kind not in ("simple", "rank", "reverse_rank"):
        raise ValueError(f"unknown reward kind: {cfg.kind!r}")

    w = cfg.beam_width
    base = cfg.decay_base if cfg.decay_base is not None else w
    correct_r = cfg.correct_reward
    wrong_r = cfg.wrong_penalty
    kind = cfg.kind
    adjust = cfg.adjust_rewards

    # Rank index i = 0 is the top beam.
    ranks = torch.arange(w, dtype=torch.float32, device=device)
    if kind == "simple":
        correct_by_rank = torch.full((w,), correct_r, device=device)
    elif kind == "rank":
        correct_by_rank = correct_r * torch.exp(-ranks / base)
    else:  # reverse_rank
        correct_by_rank = correct_r * torch.exp(ranks / base)

    def _reward(correct_mask: torch.Tensor) -> torch.Tensor:
        # Belt-and-suspenders: if an upstream bug returned < w rollouts (e.g.
        # beam search collapsed because the prompt hit block_size), pad with
        # zeros so training doesn't crash mid-run. Rollout.py already pads to
        # w; this is defensive coverage for future regressions.
        if correct_mask.shape[0] < w:
            pad = torch.zeros(w - correct_mask.shape[0],
                              dtype=correct_mask.dtype, device=correct_mask.device)
            correct_mask = torch.cat([correct_mask, pad])
        assert correct_mask.shape == (w,), (
            f"expected correct_mask of shape ({w},), got {tuple(correct_mask.shape)}"
        )
        rewards_per_rank = correct_by_rank.to(correct_mask.device)
        cm = correct_mask.float()
        rewards = cm * rewards_per_rank + (1.0 - cm) * wrong_r

        if adjust and 0 < cm.sum().item() < w:
            # Rescale wrong-reward magnitude so sum(rewards) == 0 exactly.
            # Want (num_wrong) * (-w_effective) + sum_correct = 0
            # → w_effective = sum_correct / num_wrong.
            sum_correct = (cm * rewards_per_rank).sum()
            num_wrong = (1.0 - cm).sum()
            w_effective = sum_correct / (num_wrong + 1e-12)
            rewards = cm * rewards_per_rank + (1.0 - cm) * (-w_effective)
        return rewards

    return _reward


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Per-group mean-baseline advantage.

    ``A_i = r_i - mean(r)`` — no std normalization (paper §2.3, citing
    Liu et al. 2025). Stable for binary-ish rewards where std collapses
    when all beams succeed or all fail.
    """
    return rewards - rewards.mean()
