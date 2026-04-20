"""
Appendix C / Table 4 reproduction: per-token-category probability + accuracy.

For each test example we teacher-force the ground-truth answer through the
model, collect the softmax distribution at every answer position, and group
by the *target* token's category. The headline statistic is

    P[cat]   = E[ P(target_i | prefix_i) ]         for i s.t. cat(target_i)=cat
    acc[cat] = E[ 1[argmax_i = target_i] ]

which directly reproduces the paper's "Probability / Accuracy" columns.

The ignored positions are padding (target = PAD_CHAR) and prompt positions
(masked to PAD in ``SymbolicDataset.__getitem__``).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from mingpt.dataset import SymbolicDataset
from mingpt.vocab import PAD_CHAR, build_vocab

from .token_category import CATEGORY_NAMES, TokenCategory, category_map


def _load_model(model_path: Path, vocab: list[str], block_size: int,
                n_layer: int, n_head: int, n_embd: int, device: str) -> torch.nn.Module:
    """Build a GPT with the given hyperparameters and load weights."""
    from mingpt import model as mingpt_model

    cfg = mingpt_model.GPTConfig(
        vocab_size=len(vocab), block_size=block_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    )
    net = mingpt_model.GPT(cfg)
    net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    net.to(device).eval()
    return net


@torch.no_grad()
def measure(
    model: torch.nn.Module,
    dataset: SymbolicDataset,
    *,
    n_samples: int,
    batch_size: int,
    device: str,
) -> dict:
    """
    Run teacher-forced forward over ``n_samples`` lines and return per-category
    statistics ``{cat_name: {"prob_mean","prob_sem","acc","acc_sem","n"}}``.
    """
    pad_id = dataset.stoi[PAD_CHAR]
    cat_vec = category_map(list(dataset.itos.values())).to(device)

    n = min(n_samples, len(dataset))
    sums = defaultdict(lambda: {"prob_sum": 0.0, "prob_sq": 0.0,
                                "correct": 0, "count": 0})

    for start in range(0, n, batch_size):
        xs, ys = zip(*(dataset[i] for i in range(start, min(start + batch_size, n))))
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)

        logits, _ = model(x)                             # (B, T, V)
        # SymbolicDataset.__getitem__ already returns a shifted target
        # sequence: y[:, t] is the intended label for logits[:, t, :]
        # (prompt positions are pre-masked to PAD for loss-ignoring).
        # Adding another shift here would make this "predict 2-ahead".
        targets = y                                      # (B, T)
        probs = F.softmax(logits, dim=-1)
        tgt_prob = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # P(target)
        preds = probs.argmax(dim=-1)                     # (B, T)
        correct = preds.eq(targets)

        valid = targets.ne(pad_id)                       # drop padded positions
        tgt_cat = cat_vec[targets]                       # (B, T)

        # Flatten valid positions and aggregate.
        v_cats = tgt_cat[valid].tolist()
        v_prob = tgt_prob[valid].tolist()
        v_corr = correct[valid].tolist()
        for c, p, ok in zip(v_cats, v_prob, v_corr):
            bucket = sums[c]
            bucket["prob_sum"] += p
            bucket["prob_sq"] += p * p
            bucket["correct"] += int(ok)
            bucket["count"] += 1

    out: dict[str, dict] = {}
    for cat_id, b in sorted(sums.items()):
        name = CATEGORY_NAMES.get(cat_id, f"cat_{cat_id}")
        n_i = b["count"]
        if n_i == 0:
            continue
        mean_p = b["prob_sum"] / n_i
        var_p = max(0.0, b["prob_sq"] / n_i - mean_p * mean_p)
        sem_p = math.sqrt(var_p / n_i)
        acc = b["correct"] / n_i
        sem_a = math.sqrt(acc * (1 - acc) / n_i) if n_i > 1 else 0.0
        out[name] = {
            "prob_mean": mean_p, "prob_sem": sem_p,
            "acc": acc, "acc_sem": sem_a, "n": n_i,
        }
    return out


def print_table(stats: dict) -> None:
    """Pretty-print like paper Table 4."""
    header = "  Category       n       P(target)             accuracy"
    print(header)
    print("-" * len(header))
    order = ["SIGN", "OPERATOR", "NUMBER", "VARIABLE", "DELIMITER", "PAD", "OTHER"]
    for name in order:
        if name not in stats:
            continue
        s = stats[name]
        print(f"  {name:<12} {s['n']:>7}  "
              f"{s['prob_mean']:.3f} ± {s['prob_sem']:.3f}      "
              f"{s['acc']:.3f} ± {s['acc_sem']:.3f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Reproduce paper Table 4 on a checkpoint.")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--test_corpus", required=True, type=Path)
    p.add_argument("--block_size", type=int, default=850)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--extended_vocab", action="store_true")
    p.add_argument("--max_number_token", type=int, default=101)
    p.add_argument("--n_samples", type=int, default=1000,
                   help="Number of test lines to score (paper uses 1000).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out_json", type=Path, default=None)
    args = p.parse_args(argv)

    vocab = build_vocab(extended=args.extended_vocab, max_number_token=args.max_number_token)
    data = args.test_corpus.read_text()
    dataset = SymbolicDataset(args.block_size, vocab, data, use_extended_vocab=args.extended_vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(args.model_path, vocab, args.block_size,
                        args.n_layer, args.n_head, args.n_embd, device)
    print(f"model: {args.model_path.name} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    print(f"device: {device}")
    stats = measure(model, dataset, n_samples=args.n_samples,
                    batch_size=args.batch_size, device=device)
    print_table(stats)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(stats, indent=2))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
