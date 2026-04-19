"""
Unified decoding API for polynomial-decomposition models.

This module provides ``greedy_generate`` and ``beam_search`` as the single
canonical entry points. They transparently use the KV-cache fast path when
the underlying model supports it (``GPTWithKVCache``), and fall back to the
plain autoregressive loop otherwise.

Legacy entry points kept for backwards compatibility (do not delete without
regression testing BGRPO):
    * ``utils.sample``
    * ``utils.beam_search``
    * ``utils.multi_sampling``
    * ``GPT.generate`` / ``GPT.beam_search``
    * ``GPT_hf.generate``
    * ``GPTWithKVCache.generate_with_cache`` / ``beam_search_with_cache``

New code should prefer the functions in this module. The existing tests and
BGRPO training path must keep working, so this module deliberately produces
bit-exact-equivalent outputs to the legacy beam/greedy implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    v, _ = torch.topk(logits, k)
    return torch.where(logits < v[:, [-1]], torch.full_like(logits, -float("inf")), logits)


def _model_forward_logits(model, input_ids: torch.Tensor, *, hf: bool) -> torch.Tensor:
    """Return last-step logits of shape (batch, vocab), tolerating HF and plain GPT outputs."""
    out = model(input_ids) if hf else model(input_ids)
    if hasattr(out, "logits"):
        logits = out.logits
    elif isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    return logits


@torch.no_grad()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    do_sample: bool = False,
) -> torch.Tensor:
    """
    Autoregressive generation. Appends up to ``max_new_tokens`` tokens to
    ``input_ids`` and returns the full sequence.

    Uses ``model.generate_with_cache`` when available (KV-cache fast path);
    otherwise falls back to re-running the full-context forward each step.
    """
    if hasattr(model, "generate_with_cache"):
        return model.generate_with_cache(
            input_ids,
            max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
        )

    model.eval()
    block_size = model.get_block_size()
    hf = getattr(model, "hf", False)
    x = input_ids

    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits = _model_forward_logits(model, x_cond, hf=hf)[:, -1, :] / temperature
        if top_k is not None:
            logits = _top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        x = torch.cat((x, idx_next), dim=1)

    return x


@dataclass
class Beam:
    """A beam-search hypothesis."""

    tokens: torch.Tensor  # shape: (seq_len,)
    logprob: float        # sum of log-probs (higher is better)
    length: int           # len(tokens), materialized for convenience


@torch.no_grad()
def beam_search(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    beam_width: int,
    tokenizer,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> List[Beam]:
    """
    Beam search rooted at the single prompt ``input_ids`` (shape (1, t)).

    ``tokenizer`` must expose ``PAD_INDEX``/``END_INDEX`` and ``MASK_INDEX``
    (either a ``SymbolicTokenizer`` or a ``SymbolicDataset`` works — both
    provide the attributes).

    Returns the top ``beam_width`` beams sorted by cumulative log-prob.
    Tokens that are MASK are coerced to END mid-beam, matching the legacy
    implementation in ``utils.beam_search``.
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, (
        f"beam_search expects a single prompt, got shape {tuple(input_ids.shape)}"
    )
    end_idx = getattr(tokenizer, "END_INDEX", getattr(tokenizer, "eos_token_id", None))
    mask_idx = getattr(tokenizer, "MASK_INDEX", getattr(tokenizer, "mask_token_id", None))
    if end_idx is None or mask_idx is None:
        raise ValueError("tokenizer must expose END_INDEX/MASK_INDEX (or eos/mask ids)")

    model.eval()
    block_size = model.get_block_size()
    hf = getattr(model, "hf", False)

    # Beam state: (sequence tensor (1, seq_len), list of per-step log-probs, cum log-prob).
    beams: List[tuple] = [(input_ids, [], 0.0)]

    for _ in range(max_new_tokens):
        candidates: List[tuple] = []
        for seq, logp_list, total_logp in beams:
            seq_cond = seq if seq.size(1) <= block_size else seq[:, -block_size:]
            if seq_cond[0, -1].item() == end_idx:
                candidates.append((seq, logp_list, total_logp))
                continue

            logits = _model_forward_logits(model, seq_cond, hf=hf)[:, -1, :] / temperature
            if top_k is not None:
                logits = _top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)
            for i in range(beam_width):
                next_token = topk_indices[0, i].unsqueeze(0)
                if next_token.item() == mask_idx:
                    next_token = next_token.clone()
                    next_token[0] = end_idx
                new_seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)
                step_logp = torch.log(topk_probs[0, i]).item()
                new_logp_list = logp_list + [step_logp]
                candidates.append((new_seq, new_logp_list, sum(new_logp_list)))

        candidates.sort(key=lambda b: b[2], reverse=True)
        beams = candidates[:beam_width]

        if all(b[0][0, -1].item() == end_idx for b in beams):
            break

    return [Beam(tokens=b[0][0], logprob=b[2], length=b[0].size(1)) for b in beams]


def pad_beams(beams: List[Beam], pad_value: int = 0) -> torch.Tensor:
    """Stack beams into a (num_beams, max_len) tensor padded with ``pad_value``."""
    return pad_sequence([b.tokens for b in beams], batch_first=True, padding_value=pad_value)
