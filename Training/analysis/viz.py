"""
Figures for the analysis toolkit.

* ``plot_top3`` reproduces paper Figure 9: top-3 probabilities at every
  answer position with the correct token highlighted. Exposes the sign-
  token uncertainty visually — at sign positions the top-1/top-2 gap is
  tiny, elsewhere it is large.
* ``plot_attention`` reproduces paper Figure 10: a single-head attention
  heatmap with token labels on both axes.
* ``plot_circuit_scores`` shows circuit scores as layer × head heatmaps —
  useful for picking which heads to hand-inspect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mingpt.vocab import MASK_CHAR, PAD_CHAR, build_vocab

from .circuits import (
    _encode_line,
    _load_model,
    _locate_split,
    capture_attention,
    monomial_segments,
    score_delimiter_heads,
    score_previous_token_heads,
    score_within_monomial_heads,
)
from .token_category import TokenCategory, categorize


# ---------------------------------------------------------------------------
# Fig 9 — top-3 probability strip
# ---------------------------------------------------------------------------

def plot_top3(
    model: torch.nn.Module,
    input_ids: torch.Tensor,       # (1, T)
    itos: list[str],
    out_path: Path,
    title: str | None = None,
) -> None:
    """Top-3 probability bars per answer position; correct token highlighted."""
    model.eval()
    device = next(model.parameters()).device
    x = input_ids.to(device)
    with torch.no_grad():
        logits, _ = model(x)
    probs = F.softmax(logits[0], dim=-1).cpu()            # (T, V)

    tokens = [itos[i] for i in x[0].tolist()]
    prompt_end, ans_start, ans_end = _locate_split(tokens)

    # We plot one column per answer position; the target at column t is the
    # token tokens[ans_start + t], predicted from logits[ans_start + t - 1].
    cols = []
    for t in range(ans_start, ans_end):
        p_t = probs[t - 1]
        top_prob, top_ids = torch.topk(p_t, k=3)
        cols.append((tokens[t], top_ids.tolist(), top_prob.tolist()))

    n = len(cols)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), 4))
    for col_idx, (target, ids, ps) in enumerate(cols):
        for rank, (tid, pr) in enumerate(zip(ids, ps)):
            tok = itos[tid]
            color = "red" if tok == target else "black"
            ax.text(col_idx, 3 - rank, f"{tok}\n{pr:.2f}",
                    ha="center", va="center", fontsize=7, color=color,
                    fontweight=("bold" if tok == target else "normal"))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_yticks([1, 2, 3], ["top-3", "top-2", "top-1"])
    ax.set_xticks(range(n), [c[0] for c in cols], rotation=90, fontsize=6)
    ax.set_title(title or "Top-3 predictions per answer position (red = target)")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 10 — single-head attention heatmap
# ---------------------------------------------------------------------------

def plot_attention(
    attn: torch.Tensor,             # (T, T)
    tokens: list[str],
    out_path: Path,
    title: str,
    region: slice | None = None,
) -> None:
    """Render one head's attention matrix as a heatmap."""
    if region is None:
        region = slice(0, attn.size(0))
    a = attn[region, region].numpy()
    labels = tokens[region]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.18),
                                    max(5, len(labels) * 0.18)))
    im = ax.imshow(a, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=5)
    ax.set_yticks(range(len(labels)), labels, fontsize=5)
    ax.set_xlabel("key (attended-to)")
    ax.set_ylabel("query (row)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Circuit-score heatmap (L × H)
# ---------------------------------------------------------------------------

def plot_circuit_scores(
    scores: dict[str, torch.Tensor],
    out_path: Path,
    title: str = "Circuit scores (L × H)",
) -> None:
    names = list(scores.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), squeeze=False)
    for ax, name in zip(axes[0], names):
        s = scores[name].numpy()
        im = ax.imshow(s, cmap="magma", aspect="auto",
                       vmin=0, vmax=max(0.01, float(s.max())))
        ax.set_title(name)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(s.shape[1]))
        ax.set_yticks(range(s.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI: build all three figures for one (model, line) pair
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Render paper figures 9/10 from a checkpoint.")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--test_corpus", required=True, type=Path)
    p.add_argument("--line_idx", type=int, default=0)
    p.add_argument("--block_size", type=int, default=850)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--extended_vocab", action="store_true")
    p.add_argument("--max_number_token", type=int, default=101)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--which_attn", default="previous_token",
                   choices=["previous_token", "within_monomial", "delimiter"],
                   help="Which circuit's top head to render for the Fig-10 heatmap.")
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

    # Score all three circuits
    input_segs = monomial_segments(res.tokens, 0, prompt_end)
    prev = score_previous_token_heads(res.attentions)
    wm = score_within_monomial_heads(res.attentions, input_segs,
                                     positions=slice(0, prompt_end))
    delim = score_delimiter_heads(res.attentions, res.tokens, "&",
                                  positions=slice(ans_start, ans_end))

    out = args.out_dir
    plot_circuit_scores({"prev-token": prev, "within-monomial": wm, "delimiter": delim},
                        out / "circuit_scores.png",
                        title=f"{args.model_path.name}  line_idx={args.line_idx}")

    # Fig 9
    plot_top3(model, input_ids, vocab, out / "fig9_top3.png",
              title=f"Top-3 predictions — {args.model_path.name}")

    # Fig 10: pick the top head for the chosen circuit
    score_tensor = {"previous_token": prev, "within_monomial": wm, "delimiter": delim}[args.which_attn]
    flat = score_tensor.flatten()
    best_flat = int(torch.argmax(flat))
    H = score_tensor.size(1)
    best_layer, best_head = best_flat // H, best_flat % H
    attn = res.attentions[best_layer][best_head]
    # Keep the heatmap to the prompt region by default — the paper plots the
    # input polynomial only for these figures.
    region = (slice(0, prompt_end) if args.which_attn != "delimiter"
              else slice(ans_start, ans_end))
    plot_attention(attn, res.tokens, out / f"fig10_{args.which_attn}.png",
                   title=(f"{args.which_attn} head  L{best_layer} H{best_head}  "
                          f"score={float(flat[best_flat]):.3f}"),
                   region=region)

    print(f"wrote plots to {out}")


if __name__ == "__main__":
    main()
