"""
Paper Appendix D reproduction: detectors for the "Monomial Heads" circuit.

The paper proposes a two-stage circuit inside the D2 models:

  * Layer-0 — **previous-token heads**: a few heads consistently attend to
    positions 1-5 behind the current position, acting as a short-range
    "what just happened" probe.
  * Layer-1 — **within-monomial heads**: tokens inside a monomial attend to
    earlier tokens of the *same* monomial, establishing monomial membership.
    Monomials in the input polynomial are delimited by the top-level ``+``
    operator (prefix notation).
  * Layer-1 — **delimiter heads**: on the answer side, attention concentrates
    on the ``&`` delimiter tokens that separate the outer polynomial ``g``
    from each inner polynomial ``h_i``.

This module provides a score per (layer, head) for each of those three
patterns; higher = more the head looks like the circuit. Paper Figure 10
shows the canonical exemplars; we surface them as rankings so a user can
locate the paper's heads on any checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from mingpt.vocab import MASK_CHAR, PAD_CHAR, build_vocab


@dataclass
class AttentionResult:
    """Container for a single forward pass' attention stack."""

    input_ids: torch.Tensor          # (T,) input token ids
    tokens: list[str]                # len=T, string form
    attentions: list[torch.Tensor]   # L tensors each (H, T, T)


# ---------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_attention(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    itos: dict[int, str] | list[str],
) -> AttentionResult:
    """
    Run the model once with ``return_attentions=True`` and return the per-
    layer attention tensors squeezed to (n_head, T, T).

    ``input_ids`` must be shape (1, T) — we only analyse one sequence at a
    time. The model is put into eval mode (dropout disabled) before the
    forward, otherwise paper-style deterministic heatmaps don't reproduce.
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, (
        "capture_attention expects a single sequence (1, T); got "
        f"{tuple(input_ids.shape)}")
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    out = model(input_ids, return_attentions=True)
    # Base GPT returns (logits, loss, attentions); GPT_hf returns a
    # CausalLMOutputWithPast whose .attentions holds the list.
    if isinstance(out, tuple):
        attentions = out[2]
    else:
        attentions = out.attentions

    # Each attn tensor is (1, H, T, T) → strip batch dim.
    atts = [a.squeeze(0).detach().cpu() for a in attentions]

    itos_list = (
        [itos[i] for i in range(len(itos))] if isinstance(itos, dict) else list(itos)
    )
    tokens = [itos_list[i] for i in input_ids[0].tolist()]
    return AttentionResult(
        input_ids=input_ids[0].cpu(),
        tokens=tokens,
        attentions=atts,
    )


# ---------------------------------------------------------------------------
# Segmentation helpers (prefix-notation polynomials)
# ---------------------------------------------------------------------------

def monomial_segments(tokens: list[str], start: int, end: int,
                      plus_token: str = "+") -> list[tuple[int, int]]:
    """
    Split the token range ``[start, end)`` into monomials by cutting at every
    occurrence of ``plus_token``. Each segment is [seg_start, seg_end) in
    *absolute* token positions. Empty segments are dropped.

    The paper's prefix notation uses ``+ A B`` as "A + B", so every ``+``
    marks a monomial boundary; we don't need to track arity — just slicing
    between consecutive ``+`` tokens already gives monomial membership that
    matches the "within-monomial" pattern of Fig 10.
    """
    cuts: list[int] = [start]
    for i in range(start, end):
        if tokens[i] == plus_token:
            cuts.append(i)
    cuts.append(end)
    segs: list[tuple[int, int]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            segs.append((a, b))
    return segs


def _locate_split(tokens: list[str]) -> tuple[int, int, int]:
    """Return (prompt_end, answer_start, answer_end).

    ``prompt_end`` = index of the first MASK_CHAR or '?' separator. Answer
    runs from that separator to the next MASK_CHAR (or end of sequence if
    no trailing one).
    """
    T = len(tokens)
    first_mask = next(
        (i for i, t in enumerate(tokens) if t in (MASK_CHAR, "?")),
        None,
    )
    if first_mask is None:
        return T, T, T
    ans_start = first_mask + 1
    next_mask = next(
        (i for i, t in enumerate(tokens[ans_start:], start=ans_start)
         if t in (MASK_CHAR, "?", PAD_CHAR)),
        T,
    )
    return first_mask, ans_start, next_mask


# ---------------------------------------------------------------------------
# Circuit scores
# ---------------------------------------------------------------------------

def score_previous_token_heads(
    attentions: list[torch.Tensor],
    positions: slice | None = None,
    k_min: int = 1,
    k_max: int = 5,
) -> torch.Tensor:
    """
    For every (layer, head), measure how much of each row's attention lands
    on positions ``[i-k_max, i-k_min]`` (inclusive on both ends).

    Returns a tensor of shape (L, H) with average mass in [0, 1].

    The paper's pattern (Fig 10 left) is "attend 1-5 back". A head with
    score ≈ 1 fully implements that circuit; a random head scores ~ k_range/T.
    """
    L = len(attentions)
    H = attentions[0].size(0)
    T = attentions[0].size(-1)

    if positions is None:
        positions = slice(k_max, T)     # rows 0..k_max have no valid lookback

    mask = torch.zeros(T, T)
    for i in range(T):
        lo = max(0, i - k_max)
        hi = max(0, i - k_min + 1)      # exclusive upper bound
        if hi > lo:
            mask[i, lo:hi] = 1.0

    out = torch.zeros(L, H)
    for L_idx, attn in enumerate(attentions):
        masked = (attn * mask)[:, positions, :]           # (H, rows, T)
        out[L_idx] = masked.sum(dim=(-1, -2)) / max(1, masked.size(1))
    return out


def score_within_monomial_heads(
    attentions: list[torch.Tensor],
    segments: list[tuple[int, int]],
    positions: slice | None = None,
) -> torch.Tensor:
    """
    Per (L, H): average fraction of each row's attention that lands inside
    the *same* monomial segment as the row itself. Rows outside every
    segment contribute zero. High scores indicate the paper's "within-
    monomial" Fig 10 middle pattern.
    """
    T = attentions[0].size(-1)
    seg_mask = torch.zeros(T, T)
    for a, b in segments:
        seg_mask[a:b, a:b] = 1.0

    L = len(attentions)
    H = attentions[0].size(0)

    if positions is None:
        positions = slice(0, T)
    rows_slice = seg_mask[positions]                      # (rows, T)
    rows_ok = (rows_slice.sum(dim=-1) > 0).float()        # valid rows

    out = torch.zeros(L, H)
    denom = max(1.0, rows_ok.sum().item())
    for L_idx, attn in enumerate(attentions):
        # sum of in-segment attention per (head, row)
        in_seg = (attn[:, positions, :] * rows_slice.unsqueeze(0)).sum(dim=-1)
        out[L_idx] = (in_seg * rows_ok.unsqueeze(0)).sum(dim=-1) / denom
    return out


def score_delimiter_heads(
    attentions: list[torch.Tensor],
    tokens: list[str],
    delimiter: str = "&",
    positions: slice | None = None,
) -> torch.Tensor:
    """
    Per (L, H): fraction of each row's attention mass that lands on any
    ``delimiter`` token. The paper's Fig 10 right panel is this for ``&``
    on the answer side.
    """
    T = attentions[0].size(-1)
    delim_positions = torch.zeros(T)
    for i, t in enumerate(tokens):
        if t == delimiter:
            delim_positions[i] = 1.0

    L = len(attentions)
    H = attentions[0].size(0)
    if positions is None:
        positions = slice(0, T)
    rows = attentions[0].size(-1)  # recompute isn't needed; just for clarity

    out = torch.zeros(L, H)
    n_rows = max(1, len(range(*positions.indices(T))))
    for L_idx, attn in enumerate(attentions):
        mass = (attn[:, positions, :] * delim_positions).sum(dim=-1)   # (H, rows)
        out[L_idx] = mass.sum(dim=-1) / n_rows
    return out


# ---------------------------------------------------------------------------
# Top-k reporting
# ---------------------------------------------------------------------------

def top_heads(scores: torch.Tensor, k: int = 3) -> list[tuple[int, int, float]]:
    """Return [(layer, head, score), ...] for the k highest entries."""
    L, H = scores.shape
    flat = scores.flatten()
    vals, idx = torch.topk(flat, min(k, flat.numel()))
    return [(int(i) // H, int(i) % H, float(v)) for v, i in zip(vals.tolist(), idx.tolist())]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_model(model_path: Path, vocab: list[str], block_size: int,
                n_layer: int, n_head: int, n_embd: int, device: str) -> torch.nn.Module:
    from mingpt import model as mingpt_model

    cfg = mingpt_model.GPTConfig(
        vocab_size=len(vocab), block_size=block_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    )
    net = mingpt_model.GPT(cfg)
    net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    net.to(device).eval()
    return net


def _encode_line(line: str, stoi: dict[str, int], extended: bool) -> torch.Tensor:
    """Tokenize a ``<expanded> ? <answer>`` line into a single-batch tensor."""
    if extended:
        sep = "?" if "?" in line else MASK_CHAR
    else:
        sep = MASK_CHAR
        line = line.replace("?", MASK_CHAR)
    parts = line.split(sep)
    assert len(parts) >= 2, f"malformed line: {line[:60]}..."
    inp_tokens = [t for t in parts[0].split() if t]
    ans_tokens = [t for t in parts[1].split() if t]
    all_tokens = inp_tokens + [MASK_CHAR] + ans_tokens + [MASK_CHAR]
    ids = [stoi[t] for t in all_tokens]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Detect paper Appendix-D circuits on a checkpoint.")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--test_corpus", required=True, type=Path,
                   help="One polynomial per line; first is used unless --line_idx is set.")
    p.add_argument("--line_idx", type=int, default=0)
    p.add_argument("--block_size", type=int, default=850)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--extended_vocab", action="store_true")
    p.add_argument("--max_number_token", type=int, default=101)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--out_json", type=Path, default=None)
    args = p.parse_args(argv)

    vocab = build_vocab(extended=args.extended_vocab, max_number_token=args.max_number_token)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(args.model_path, vocab, args.block_size,
                        args.n_layer, args.n_head, args.n_embd, device)

    lines = [l for l in args.test_corpus.read_text().splitlines() if l]
    line = lines[args.line_idx]
    input_ids = _encode_line(line, stoi, args.extended_vocab)
    res = capture_attention(model, input_ids, vocab)

    prompt_end, ans_start, ans_end = _locate_split(res.tokens)
    input_segs = monomial_segments(res.tokens, 0, prompt_end)
    # "within-monomial" on the answer side only counts monomials inside each
    # inner polynomial (between '&' delimiters). For simplicity we still split
    # by '+', which matches paper Fig 10 behavior on either side.
    answer_segs = monomial_segments(res.tokens, ans_start, ans_end)

    prev = score_previous_token_heads(res.attentions)
    in_mono = score_within_monomial_heads(res.attentions, input_segs,
                                          positions=slice(0, prompt_end))
    delim = score_delimiter_heads(res.attentions, res.tokens, delimiter="&",
                                  positions=slice(ans_start, ans_end))

    summary = {
        "prompt_len": prompt_end,
        "answer_len": ans_end - ans_start,
        "n_input_monomials": len(input_segs),
        "n_answer_monomials": len(answer_segs),
        "previous_token_top": top_heads(prev, args.top_k),
        "within_monomial_top": top_heads(in_mono, args.top_k),
        "delimiter_top": top_heads(delim, args.top_k),
    }
    print(json.dumps(summary, indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            **summary,
            "previous_token_all": prev.tolist(),
            "within_monomial_all": in_mono.tolist(),
            "delimiter_all": delim.tolist(),
        }, indent=2))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
