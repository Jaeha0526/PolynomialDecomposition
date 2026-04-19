"""
PPO-clipped objective + KL regularizer for BGRPO.

Given per-token log-probs under three policies (π_θ current with gradients,
π_θold at rollout time, π_ref frozen), advantages per beam, and a mask
identifying which tokens are real (non-pad) completion tokens, computes
the scalar loss the trainer backprops through.

Loss:
    L(θ) = -E_i,t [ min(ρ · A_i, clip(ρ, 1-ε, 1+ε) · A_i) ]
              + β · D_KL(π_θ || π_ref)

where:
    ρ = exp(logp_policy_t - logp_rollout_t)      per-token
    A_i = per-beam advantage (broadcast to tokens via the mask)
    D_KL uses Schulman k1 estimator:
        π_ref/π_θ - log(π_ref/π_θ) - 1
        = exp(logp_ref - logp_policy) - (logp_ref - logp_policy) - 1

All terms are masked and averaged over real completion tokens. The result
is a single scalar the optimizer minimizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class ObjectiveConfig:
    clip_epsilon: float = 0.2   # PPO clip parameter (paper: 0.2)
    kl_beta: float = 0.01       # KL regularization coefficient (paper: 0.01)


class GRPOObjective:
    """Stateless; holds only the config. Call ``compute(...)`` per step."""

    def __init__(self, cfg: ObjectiveConfig) -> None:
        self.cfg = cfg

    def compute(
        self,
        logprobs_policy: torch.Tensor,    # (w, T), requires_grad=True
        logprobs_rollout: torch.Tensor,   # (w, T), detached — π_θold
        logprobs_reference: torch.Tensor, # (w, T), detached — π_ref
        advantages: torch.Tensor,         # (w,)
        completion_mask: torch.Tensor,    # (w, T) bool/float, 1 for real tokens
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns (loss, telemetry). ``telemetry`` is plain Python floats
        suitable for wandb (no tensors, no graph).
        """
        cfg = self.cfg
        mask = completion_mask.float()
        n_tokens = mask.sum().clamp_min(1.0)

        # Per-token importance ratio ρ = exp(Δ log-prob).
        log_ratio = logprobs_policy - logprobs_rollout
        ratio = torch.exp(log_ratio)

        # Broadcast per-beam advantages to per-token.
        adv_per_token = advantages.unsqueeze(1).expand_as(ratio)

        unclipped = ratio * adv_per_token
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * adv_per_token
        per_token_pg = -torch.min(unclipped, clipped)   # we minimize, so negate

        # Schulman k1 KL estimator: exp(x) - x - 1 where x = logp_ref - logp_policy
        kl_arg = logprobs_reference - logprobs_policy
        per_token_kl = torch.exp(kl_arg) - kl_arg - 1.0

        per_token_loss = per_token_pg + cfg.kl_beta * per_token_kl
        loss = (per_token_loss * mask).sum() / n_tokens

        # Telemetry — all computed once, detached, .item()'d for wandb.
        # ``clip_frac`` follows the TRL / stable-baselines convention:
        # fraction of tokens where the ratio was outside the trust region
        # |ρ - 1| > ε, regardless of advantage sign.
        with torch.no_grad():
            outside_trust_region = (torch.abs(ratio - 1.0) > cfg.clip_epsilon).float()
            clip_frac = (outside_trust_region * mask).sum() / n_tokens
            mean_ratio = (ratio * mask).sum() / n_tokens
            mean_kl = (per_token_kl * mask).sum() / n_tokens
            mean_pg = (per_token_pg * mask).sum() / n_tokens
        telemetry = {
            "bgrpo/loss": float(loss.item()),
            "bgrpo/pg_loss": float(mean_pg.item()),
            "bgrpo/kl": float(mean_kl.item()),
            "bgrpo/clip_frac": float(clip_frac.item()),
            "bgrpo/mean_ratio": float(mean_ratio.item()),
            "bgrpo/adv_mean": float(advantages.mean().item()),
            "bgrpo/adv_std": float(advantages.std(unbiased=False).item()),
            "bgrpo/n_tokens": float(n_tokens.item()),
        }
        return loss, telemetry


def gather_token_logprobs(
    logits: torch.Tensor,   # (B, T, V)
    target_ids: torch.Tensor,  # (B, T)
) -> torch.Tensor:
    """
    Extract the log-prob of the ``target_ids[b, t]`` token under
    ``logits[b, t]``. Shape (B, T).

    Used by the trainer to re-score completion tokens under π_θ and π_ref.
    The model must have been called with inputs that produce logits at the
    positions ``target_ids`` occupies (standard teacher-forcing alignment:
    logits at position t-1 predict token at t).
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
