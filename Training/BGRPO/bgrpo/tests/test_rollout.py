"""Unit tests for BeamRollout + SamplingRollout shape/behavior contracts."""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO / "Training") not in sys.path:
    sys.path.insert(0, str(REPO / "Training"))

from mingpt import model as mingpt_model  # noqa: E402
from mingpt.tokenizer import SymbolicTokenizer  # noqa: E402
from mingpt.vocab import build_simple_vocab  # noqa: E402

from bgrpo.rollout import BeamRollout, SamplingRollout  # noqa: E402


def _tiny_model():
    vocab = build_simple_vocab()
    tok = SymbolicTokenizer(vocab)
    cfg = mingpt_model.GPTConfig(
        vocab_size=len(vocab), block_size=32, n_layer=1, n_head=2, n_embd=16,
    )
    m = mingpt_model.GPT(cfg)
    return m, tok


def test_sampling_rollout_shape_contract():
    torch.manual_seed(0)
    m, tok = _tiny_model()
    rollout = SamplingRollout(tok, num_samples=4, max_new_tokens=6)
    prompt = torch.tensor([[tok.stoi["a"], tok.stoi["0"]]], dtype=torch.long)
    ro = rollout(m, prompt)
    assert ro.completion_ids.shape[0] == 4
    assert ro.completion_mask.shape == ro.completion_ids.shape
    assert ro.rollout_logprobs.shape == ro.completion_ids.shape
    assert len(ro.beam_strings) == 4
    assert ro.beam_ranks.tolist() == [0, 1, 2, 3]


def test_beam_and_sampling_agree_on_api():
    torch.manual_seed(1)
    m, tok = _tiny_model()
    prompt = torch.tensor([[tok.stoi["a"], tok.stoi["0"]]], dtype=torch.long)

    beam = BeamRollout(tok, beam_width=4, max_new_tokens=5)(m, prompt)
    samp = SamplingRollout(tok, num_samples=4, max_new_tokens=5)(m, prompt)

    # Same RolloutResult dataclass, same field set, same number of outputs.
    assert set(beam.__dataclass_fields__) == set(samp.__dataclass_fields__)
    assert beam.completion_ids.shape[0] == samp.completion_ids.shape[0] == 4
    # Rollout log-probs are masked to valid positions in both.
    assert (beam.rollout_logprobs * (~beam.completion_mask).float()).abs().sum().item() == 0.0
    assert (samp.rollout_logprobs * (~samp.completion_mask).float()).abs().sum().item() == 0.0


def test_sampling_logprobs_match_retrace():
    """Rollout-time log-probs should match log_softmax+gather at retrace."""
    torch.manual_seed(2)
    m, tok = _tiny_model()
    m.eval()  # dropout off, or retrace disagrees with rollout
    prompt = torch.tensor([[tok.stoi["a"]]], dtype=torch.long)
    ro = SamplingRollout(tok, num_samples=3, max_new_tokens=4, temperature=1.0)(m, prompt)
    if ro.completion_ids.numel() == 0:
        return

    from bgrpo._utils import forward_logits
    from bgrpo.objective import gather_token_logprobs

    full = torch.cat([prompt.expand(3, -1), ro.completion_ids], dim=1)
    comp_len = ro.completion_ids.size(1)
    prompt_len = prompt.size(1)
    with torch.no_grad():
        logits, _ = forward_logits(m, full)
    retrace_logp = gather_token_logprobs(
        logits[:, prompt_len - 1 : prompt_len - 1 + comp_len, :], ro.completion_ids
    )
    mask = ro.completion_mask.float()
    diff = (ro.rollout_logprobs - retrace_logp) * mask
    assert diff.abs().max().item() < 1e-4, diff.abs().max().item()


def test_sampling_terminates_on_end_token():
    """If we force the model to always output end_token via a hacked forward,
    the rollout should stop after one step."""
    torch.manual_seed(3)
    m, tok = _tiny_model()
    end_id = tok.eos_token_id

    # Hack: patch the model's forward to return logits that strongly prefer end_id.
    orig_fwd = m.forward

    def biased_fwd(idx, *a, **kw):
        logits, loss = orig_fwd(idx, *a, **kw)
        biased = torch.full_like(logits, -1e9)
        biased[..., end_id] = 0.0
        return biased, loss

    m.forward = biased_fwd
    try:
        ro = SamplingRollout(tok, num_samples=3, max_new_tokens=5)(m, torch.tensor([[tok.stoi["a"]]]))
    finally:
        m.forward = orig_fwd

    # First token of every sample should be end_id.
    assert ro.completion_ids[:, 0].eq(end_id).all()
    # Subsequent tokens (if any) should be masked out.
    if ro.completion_ids.size(1) > 1:
        assert ro.completion_mask[:, 1:].eq(False).all()
