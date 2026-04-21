"""
Rollout strategies for BGRPO / GRPO.

Produces the experience the PPO objective consumes: a batch of completion
sequences plus per-token log-probs under the rollout policy (π_θold). The
trainer re-scores those same positions under the current policy (π_θ, with
gradients) and the frozen reference policy (π_ref) at optimization time.

Two strategies share the same ``RolloutResult`` output so the trainer
doesn't care which produced the experience:

  * :class:`BeamRollout`     — deterministic top-w beam search (BGRPO).
  * :class:`SamplingRollout` — iid multinomial samples from π_θold (GRPO).

The ``use_beam`` config flag on the trainer picks between them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

# ``mingpt.utils.top_k_logits`` lives in the sibling package; make sure it's
# importable whether bgrpo is run as a script or as an installed package.
_TRAINING = Path(__file__).resolve().parents[2]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))
from mingpt.utils import top_k_logits  # noqa: E402

from ._utils import forward_logits


@dataclass
class RolloutResult:
    """One beam rollout's worth of experience, for a single prompt.

    All tensors are on the same device as the input prompt. The trainer
    will call ``.detach()`` before using these as the π_θold baseline —
    gradients must not flow through rollout.
    """
    prompt_ids: torch.Tensor        # (1, prompt_len)
    completion_ids: torch.Tensor    # (beam_width, max_completion_len)
    completion_mask: torch.Tensor   # (beam_width, max_completion_len), 1 for real tokens
    rollout_logprobs: torch.Tensor  # (beam_width, max_completion_len)
    cumulative_logprobs: torch.Tensor  # (beam_width,) sum along valid positions
    beam_strings: List[str]         # decoded completions, for reward scoring
    beam_ranks: torch.Tensor        # (num_outputs,) — meaningful for beam only


class SamplingRollout:
    """
    Multinomial sampling rollout — this is what turns ``BGRPOTrainer`` into
    a vanilla GRPO trainer. Produces ``num_samples`` iid completions from
    π_θold by batched multinomial sampling at each step.

    Differences vs :class:`BeamRollout`:
      * No ranking — ``beam_ranks`` is set to ``arange(num_samples)`` for
        API compatibility but carries no ordering meaning. A ``rank`` reward
        is therefore disallowed upstream in the trainer.
      * One batched forward per step covers all samples, vs ``beam_width``
        sequential forwards for beam. At num_samples=32 / max_new_tokens=150
        this is dramatically faster than the current BeamRollout.
      * Early termination happens per-sample via the ``active`` mask; the
        batched forward still runs all samples but their post-END tokens
        are mask-zeroed and don't contribute to loss.
    """

    def __init__(
        self,
        tokenizer,
        num_samples: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> None:
        assert num_samples >= 1
        assert max_new_tokens >= 1
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

    @torch.no_grad()
    def __call__(self, model, prompt_ids: torch.Tensor) -> RolloutResult:
        assert prompt_ids.dim() == 2 and prompt_ids.size(0) == 1
        device = prompt_ids.device
        end_idx = self.tokenizer.eos_token_id
        mask_idx = self.tokenizer.mask_token_id
        pad_idx = self.tokenizer.pad_token_id
        block_size = model.get_block_size()
        n = self.num_samples
        prompt_len = prompt_ids.size(1)

        was_training = model.training
        model.eval()
        try:
            sequences = prompt_ids.expand(n, -1).contiguous().clone()
            per_step_logp: list[torch.Tensor] = []
            per_step_mask: list[torch.Tensor] = []
            active = torch.ones(n, dtype=torch.bool, device=device)

            for _ in range(self.max_new_tokens):
                if sequences.size(1) >= block_size or not active.any():
                    break
                seq_cond = sequences if sequences.size(1) <= block_size else sequences[:, -block_size:]
                logits, _ = forward_logits(model, seq_cond)
                last_logits = logits[:, -1, :] / self.temperature
                if self.top_k is not None:
                    last_logits = top_k_logits(last_logits, self.top_k)

                log_probs = F.log_softmax(last_logits, dim=-1)
                next_tokens = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)  # (n,)
                # MASK mid-sequence terminates the sample, same convention as beam.
                next_tokens = torch.where(
                    next_tokens == mask_idx,
                    torch.full_like(next_tokens, end_idx),
                    next_tokens,
                )
                step_logp = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

                # Record mask + logp BEFORE updating ``active``; the token
                # emitted at this step counts toward the sample iff the
                # sample was active coming into the step.
                per_step_mask.append(active.clone())
                per_step_logp.append(torch.where(active, step_logp, torch.zeros_like(step_logp)))

                # For inactive samples, overwrite the sampled token with PAD —
                # harmless because the mask will discard it, cleaner to read.
                token_col = torch.where(active, next_tokens, torch.full_like(next_tokens, pad_idx))
                sequences = torch.cat([sequences, token_col.unsqueeze(-1)], dim=1)
                active = active & (next_tokens != end_idx)
        finally:
            if was_training:
                model.train()

        if not per_step_logp:
            completion_ids = torch.empty(n, 0, dtype=torch.long, device=device)
            completion_mask = torch.empty(n, 0, dtype=torch.bool, device=device)
            rollout_logprobs = torch.empty(n, 0, device=device)
        else:
            completion_ids = sequences[:, prompt_len:]
            rollout_logprobs = torch.stack(per_step_logp, dim=1)
            completion_mask = torch.stack(per_step_mask, dim=1)

        cumulative = rollout_logprobs.sum(dim=1)
        beam_strings = self.tokenizer.batch_decode(completion_ids.cpu().tolist())
        return RolloutResult(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            rollout_logprobs=rollout_logprobs,
            cumulative_logprobs=cumulative,
            beam_strings=beam_strings,
            beam_ranks=torch.arange(n, device=device),
        )


class BeamRollout:
    """
    Callable-style rollout worker.

    Holds configuration (beam width, max new tokens, temperature) and a
    reference to the tokenizer. The model is passed in per-call so the
    trainer can swap between training mode and no-grad mode if it ever
    needs to.
    """

    def __init__(
        self,
        tokenizer,
        beam_width: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> None:
        assert beam_width >= 1, f"beam_width must be >= 1, got {beam_width}"
        assert max_new_tokens >= 1
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

    @torch.no_grad()
    def __call__(self, model, prompt_ids: torch.Tensor) -> RolloutResult:
        """
        Run beam search on ``prompt_ids`` (shape (1, prompt_len)) and return
        the experience. Model is assumed to expose either the plain
        ``forward(input_ids) -> (logits, loss)`` signature or the KV-cache
        path via ``forward(..., past_key_values=...)`` — we use the plain
        path here since beam search already recomputes per step, and the
        extra complexity of per-beam cache management isn't worth it for
        RL rollouts (beams are only `w` of them, ~10–32).
        """
        assert prompt_ids.dim() == 2 and prompt_ids.size(0) == 1, (
            f"expected (1, T) prompt, got {tuple(prompt_ids.shape)}"
        )
        device = prompt_ids.device
        end_idx = self.tokenizer.eos_token_id
        mask_idx = self.tokenizer.mask_token_id
        block_size = model.get_block_size()

        w = self.beam_width
        was_training = model.training
        model.eval()
        try:
            beams = self._run_beams(model, prompt_ids, w, mask_idx, end_idx, block_size, device)
        finally:
            if was_training:
                model.train()
        return self._pack(prompt_ids, beams, device)

    def _run_beams(self, model, prompt_ids, w, mask_idx, end_idx, block_size, device):
        # Each beam is (tokens, per-step-logprob list, cum_logp, finished).
        beams: list[tuple[torch.Tensor, list[float], float, bool]] = [
            (prompt_ids, [], 0.0, False),
        ]

        for _ in range(self.max_new_tokens):
            if all(b[3] for b in beams):
                break

            candidates: list[tuple[torch.Tensor, list[float], float, bool]] = []
            for seq, logp_list, total_logp, finished in beams:
                if finished or seq.size(1) >= block_size:
                    candidates.append((seq, logp_list, total_logp, True))
                    continue

                seq_cond = seq if seq.size(1) <= block_size else seq[:, -block_size:]
                logits, _ = forward_logits(model, seq_cond)
                last_logits = logits[:, -1, :] / self.temperature
                if self.top_k is not None:
                    last_logits = top_k_logits(last_logits, self.top_k)

                log_probs = F.log_softmax(last_logits, dim=-1)  # (1, vocab)
                topk_logp, topk_idx = torch.topk(log_probs, k=w, dim=-1)

                for i in range(w):
                    nxt = topk_idx[0, i]
                    step_logp = float(topk_logp[0, i].item())
                    # A beam emitting MASK mid-sequence is treated as
                    # terminated: the MASK token is the prompt/answer
                    # boundary in this vocab, not a legal output token.
                    if int(nxt.item()) == mask_idx:
                        nxt = torch.tensor(end_idx, device=device, dtype=seq.dtype)
                    new_seq = torch.cat([seq, nxt.view(1, 1)], dim=1)
                    new_finished = int(nxt.item()) == end_idx
                    candidates.append((
                        new_seq,
                        logp_list + [step_logp],
                        total_logp + step_logp,
                        new_finished,
                    ))

            candidates.sort(key=lambda b: b[2], reverse=True)
            beams = candidates[:w]
        # Pad to w if the search collapsed (can happen when the prompt is
        # already at block_size so the seed beam is marked finished before it
        # expands — produces a single-beam result that downstream rank-reward
        # shape asserts reject). Duplicate the top beam with a tiny logp
        # penalty so ranking stays stable.
        if len(beams) < w:
            pad_seed = beams[0] if beams else (
                prompt_ids, [], 0.0, True,
            )
            seq, logp_list, total_logp, _finished = pad_seed
            beams = list(beams) + [
                (seq, logp_list, total_logp - 1e-9 * (i + 1), True)
                for i in range(w - len(beams))
            ]
        return beams

    # --- packing helpers -----------------------------------------------------

    def _pack(
        self,
        prompt_ids: torch.Tensor,
        beams: list[tuple[torch.Tensor, list[float], float, bool]],
        device: torch.device,
    ) -> RolloutResult:
        """Pack the variable-length beams into padded tensors + metadata."""
        prompt_len = prompt_ids.size(1)
        # Each beam's sequence is prompt_ids + its completion tokens.
        completions = [b[0][0, prompt_len:] for b in beams]
        max_len = max((c.size(0) for c in completions), default=0)
        pad_token = self.tokenizer.pad_token_id

        if max_len == 0:
            # No new tokens produced (edge case — prompt already at block_size).
            completion_ids = torch.empty(len(beams), 0, dtype=torch.long, device=device)
            completion_mask = torch.empty(len(beams), 0, dtype=torch.bool, device=device)
            rollout_logprobs = torch.empty(len(beams), 0, device=device)
        else:
            completion_ids = torch.full(
                (len(beams), max_len), pad_token, dtype=torch.long, device=device,
            )
            completion_mask = torch.zeros(
                (len(beams), max_len), dtype=torch.bool, device=device,
            )
            rollout_logprobs = torch.zeros(
                (len(beams), max_len), dtype=torch.float, device=device,
            )
            for i, (c, (_, logp_list, _, _)) in enumerate(zip(completions, beams)):
                n = c.size(0)
                completion_ids[i, :n] = c
                completion_mask[i, :n] = True
                if logp_list:
                    rollout_logprobs[i, :len(logp_list)] = torch.tensor(
                        logp_list, device=device,
                    )

        cumulative = torch.tensor([b[2] for b in beams], device=device)
        # Decode on CPU — batch_decode iterates token-by-token and triggers a
        # CUDA sync per call otherwise.
        beam_strings = self.tokenizer.batch_decode(completion_ids.cpu().tolist())
        return RolloutResult(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            rollout_logprobs=rollout_logprobs,
            cumulative_logprobs=cumulative,
            beam_strings=beam_strings,
            beam_ranks=torch.arange(len(beams), device=device),
        )


