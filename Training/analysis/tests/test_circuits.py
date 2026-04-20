"""Unit tests for circuit-score helpers."""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "Training") not in sys.path:
    sys.path.insert(0, str(REPO / "Training"))

from analysis.circuits import (  # noqa: E402
    monomial_segments,
    score_delimiter_heads,
    score_previous_token_heads,
    score_within_monomial_heads,
    top_heads,
)


def test_monomial_segments_basic():
    tokens = ["+", "a", "b", "+", "c", "d", "e"]
    segs = monomial_segments(tokens, 0, len(tokens))
    # cuts at 0, 3 (second '+'), end → segments (0,3) and (3,7)
    assert segs == [(0, 3), (3, 7)]


def test_monomial_segments_no_plus_returns_one():
    tokens = ["a", "b", "c"]
    assert monomial_segments(tokens, 0, 3) == [(0, 3)]


def test_previous_token_score_identity_head():
    """Synthesise an attention tensor that always attends to position-1; score
    should be close to 1.0 for rows where a lookback exists."""
    T, H, L = 10, 2, 2
    attn = torch.zeros(H, T, T)
    for i in range(1, T):
        attn[:, i, i - 1] = 1.0
    attn[:, 0, 0] = 1.0      # row 0 self-attends (no prior)
    score = score_previous_token_heads([attn.clone() for _ in range(L)],
                                       k_min=1, k_max=5)
    assert score.shape == (L, H)
    assert score.min().item() > 0.9


def test_within_monomial_score():
    """Head that attends only within a segment should score 1.0 for rows
    inside that segment."""
    T, H = 6, 1
    attn = torch.zeros(H, T, T)
    # Segment (0,3): rows 0..2 attend inside 0..2 uniformly
    for i in range(3):
        attn[0, i, 0:3] = 1.0 / 3
    # Segment (3,6): rows 3..5 attend inside 3..5 uniformly
    for i in range(3, 6):
        attn[0, i, 3:6] = 1.0 / 3
    segs = [(0, 3), (3, 6)]
    score = score_within_monomial_heads([attn], segs)
    assert score.shape == (1, 1)
    assert score.item() > 0.99


def test_delimiter_score():
    T, H = 6, 1
    tokens = ["a", "b", "&", "c", "d", "&"]
    attn = torch.zeros(H, T, T)
    # Every row puts all mass on the '&' at pos 5
    attn[0, :, 5] = 1.0
    score = score_delimiter_heads([attn], tokens, delimiter="&")
    assert score.item() > 0.99


def test_top_heads_sorted():
    s = torch.tensor([[0.1, 0.4], [0.3, 0.2]])
    top = top_heads(s, k=2)
    assert top[0][2] >= top[1][2]
    assert top[0][:2] == (0, 1)  # 0.4 at (L=0, H=1)
