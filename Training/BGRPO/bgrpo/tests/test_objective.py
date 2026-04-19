"""Unit tests for bgrpo.objective."""

import torch

from bgrpo.objective import GRPOObjective, ObjectiveConfig, gather_token_logprobs


def _synthetic(w=4, t=5, vocab=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    logp_policy = torch.randn(w, t, generator=g, requires_grad=True)
    logp_rollout = logp_policy.detach().clone()
    logp_reference = torch.randn(w, t, generator=g)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])[:w]
    mask = torch.ones(w, t)
    return logp_policy, logp_rollout, logp_reference, advantages, mask


def test_loss_is_scalar_and_backpropable():
    obj = GRPOObjective(ObjectiveConfig())
    lp_p, lp_old, lp_ref, adv, m = _synthetic()
    loss, tele = obj.compute(lp_p, lp_old, lp_ref, adv, m)
    assert loss.dim() == 0
    loss.backward()
    assert lp_p.grad is not None and lp_p.grad.abs().sum().item() > 0
    assert set(tele.keys()) >= {"bgrpo/loss", "bgrpo/kl", "bgrpo/clip_frac"}


def test_zero_advantage_gives_zero_pg_loss():
    obj = GRPOObjective(ObjectiveConfig(kl_beta=0.0))  # disable KL for this test
    lp_p, lp_old, lp_ref, _, m = _synthetic()
    adv = torch.zeros(lp_p.size(0))
    loss, tele = obj.compute(lp_p, lp_old, lp_ref, adv, m)
    # No policy gradient signal, no KL penalty → loss should be exactly 0.
    assert abs(float(loss.item())) < 1e-6


def test_kl_is_zero_when_policy_equals_reference():
    obj = GRPOObjective(ObjectiveConfig(kl_beta=1.0, clip_epsilon=0.2))
    lp_p, lp_old, _, adv, m = _synthetic()
    # Force reference == policy.
    lp_ref = lp_p.detach().clone()
    _, tele = obj.compute(lp_p, lp_old, lp_ref, adv, m)
    assert abs(tele["bgrpo/kl"]) < 1e-6


def test_clip_bites_when_ratio_is_far_from_one():
    obj = GRPOObjective(ObjectiveConfig(clip_epsilon=0.2, kl_beta=0.0))
    w, t = 2, 3
    # Policy log-probs drift far from rollout so ratio >> 1+eps.
    lp_p = torch.zeros(w, t, requires_grad=True)
    lp_old = torch.full((w, t), -3.0)  # ratio = e^3 ≈ 20 ≫ 1.2
    lp_ref = torch.zeros(w, t)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(w, t)
    _, tele = obj.compute(lp_p, lp_old, lp_ref, adv, mask)
    # Every token should be clipped.
    assert tele["bgrpo/clip_frac"] > 0.9


def test_mask_is_respected():
    obj = GRPOObjective(ObjectiveConfig(kl_beta=1.0))
    w, t = 2, 4
    lp_p = torch.zeros(w, t, requires_grad=True)
    lp_old = torch.zeros(w, t)
    lp_ref = torch.full((w, t), -10.0)  # very different from policy → big KL if counted
    adv = torch.ones(w)
    # Mask out everything — loss should be 0 and n_tokens clamped to 1.
    mask = torch.zeros(w, t)
    loss, tele = obj.compute(lp_p, lp_old, lp_ref, adv, mask)
    assert abs(float(loss.item())) < 1e-6
    assert tele["bgrpo/n_tokens"] == 1.0  # clamp_min(1)


def test_gather_token_logprobs_shape_and_value():
    logits = torch.tensor([
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
    ])
    target_ids = torch.tensor([[2, 0]])
    out = gather_token_logprobs(logits, target_ids)
    assert out.shape == (1, 2)
    # Row 0 target=2, row 1 target=0, uniform logits give log(1/3).
    import math
    expected_0 = math.log(math.exp(3.0) / (math.exp(1.0) + math.exp(2.0) + math.exp(3.0)))
    expected_1 = math.log(1.0 / 3.0)
    assert abs(out[0, 0].item() - expected_0) < 1e-6
    assert abs(out[0, 1].item() - expected_1) < 1e-6
