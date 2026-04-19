"""Unit tests for bgrpo.reward."""

import math

import torch

from bgrpo.reward import RewardConfig, build_reward, compute_advantages


def _bool_tensor(values):
    return torch.tensor(values, dtype=torch.float32)


def test_simple_reward_binary():
    fn = build_reward(RewardConfig(kind="simple", beam_width=4))
    r = fn(_bool_tensor([1, 0, 1, 0]))
    assert torch.allclose(r, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_simple_reward_with_penalty():
    fn = build_reward(RewardConfig(kind="simple", beam_width=4, wrong_penalty=-0.1))
    r = fn(_bool_tensor([1, 0, 1, 0]))
    assert torch.allclose(r, torch.tensor([1.0, -0.1, 1.0, -0.1]))


def test_rank_decay_decays_with_rank():
    w = 8
    fn = build_reward(RewardConfig(kind="rank", beam_width=w, decay_base=w))
    r = fn(torch.ones(w))  # all correct
    # Top beam gets exp(0)=1; last beam gets exp(-(w-1)/w).
    expected = torch.tensor([math.exp(-i / w) for i in range(w)])
    assert torch.allclose(r, expected, atol=1e-6)


def test_rank_decay_respects_decay_base_override():
    fn = build_reward(RewardConfig(kind="rank", beam_width=8, decay_base=4))
    r = fn(torch.ones(8))
    # Steeper decay — top=1, bottom=exp(-7/4) ≈ 0.1738.
    assert abs(r[0].item() - 1.0) < 1e-6
    assert abs(r[-1].item() - math.exp(-7 / 4)) < 1e-6


def test_reverse_rank_is_monotonically_increasing_with_rank():
    fn = build_reward(RewardConfig(kind="reverse_rank", beam_width=8, decay_base=8))
    r = fn(torch.ones(8))
    # r[i] = exp(i/8); should increase with i.
    assert torch.all(r[1:] > r[:-1])


def test_adjust_rewards_yields_zero_sum():
    fn = build_reward(RewardConfig(
        kind="simple", beam_width=8, wrong_penalty=-0.1, adjust_rewards=True,
    ))
    r = fn(_bool_tensor([1, 1, 0, 0, 0, 0, 0, 0]))
    assert abs(float(r.sum().item())) < 1e-6


def test_adjust_rewards_noop_when_all_correct():
    fn = build_reward(RewardConfig(kind="simple", beam_width=4, adjust_rewards=True))
    r = fn(torch.ones(4))
    assert torch.allclose(r, torch.ones(4))


def test_compute_advantages_zero_mean():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = compute_advantages(rewards)
    assert abs(float(adv.sum().item())) < 1e-6
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, 0.5, -0.5]))


def test_build_reward_rejects_unknown_kind():
    try:
        build_reward(RewardConfig(kind="banana", beam_width=4))
    except ValueError as e:
        assert "banana" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown kind")


def test_reward_shape_assertion():
    fn = build_reward(RewardConfig(kind="simple", beam_width=4))
    try:
        fn(torch.ones(3))  # wrong shape
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError on mismatched shape")
