"""Plot token-category + circuit analysis across s200 checkpoints.

Reads per-checkpoint analysis JSON from
    data_storage/things_on_paper/BGRPO/runs/<tag>/analysis/<ckpt>/{confusion,circuits}.json

Emits one figure per run under
    data_storage/things_on_paper/BGRPO/analysis_<tag>.png

with two rows:
  - top row: per-category accuracy (left) and prob_mean (right) across steps
  - bottom row: top-1 attention-head score for each circuit type (prev-token,
    within-monomial, delimiter), annotated with the (layer, head) that won
    at each step.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/resnick/groups/Hippo/jaeha/PolynomialDecomposition")
BGRPO_ROOT = REPO / "data_storage/things_on_paper/BGRPO"

RUNS = [
    ("grpo_s200_d256",       "vanilla GRPO"),
    ("bgrpo_s200_d256",      "BGRPO (binary)"),
    ("bgrpo_rank_s200_d256", "BGRPO (rank-aware)"),
]

CATEGORIES = [
    ("SIGN",      "sign",   "#d62728"),
    ("OPERATOR",  "op",     "#1f77b4"),
    ("DELIMITER", "delim",  "#2ca02c"),
    ("NUMBER",    "num",    "#ff7f0e"),
    ("VARIABLE",  "var",    "#9467bd"),
]

CIRCUITS = [
    ("previous_token_top", "prev-token",      "#1f77b4"),
    ("within_monomial_top", "within-monomial", "#2ca02c"),
    ("delimiter_top",       "delimiter",       "#d62728"),
]

FINAL_STEP = 200


def load_run_analysis(run_tag: str) -> dict[int, dict]:
    """{step: {'confusion': ..., 'circuits': ...}} for every ckpt with both files."""
    out: dict[int, dict] = {}
    root = BGRPO_ROOT / "runs" / run_tag / "analysis"
    if not root.exists():
        return out
    for d in sorted(root.glob("checkpoint-*")):
        name = d.name.removeprefix("checkpoint-")
        step = FINAL_STEP if name == "final" else int(name)
        conf_p, circ_p = d / "confusion.json", d / "circuits.json"
        if not (conf_p.exists() and circ_p.exists()):
            continue
        out[step] = {
            "confusion": json.loads(conf_p.read_text()),
            "circuits": json.loads(circ_p.read_text()),
        }
    return out


def plot_run(run_tag: str, label: str) -> None:
    data = load_run_analysis(run_tag)
    if not data:
        print(f"  {run_tag}: no analysis JSONs yet, skipping")
        return
    steps = sorted(data)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    (ax_acc, ax_prob, ax_empty) = axes[0]
    ax_empty.axis("off")
    circuit_axes = axes[1]

    # --- top row: confusion (acc + prob) per token category ------------------
    for cat_key, short, color in CATEGORIES:
        accs  = [data[s]["confusion"].get(cat_key, {}).get("acc")       for s in steps]
        probs = [data[s]["confusion"].get(cat_key, {}).get("prob_mean") for s in steps]
        ax_acc.plot(steps,  accs,  "-o", color=color, linewidth=2, markersize=5, label=short)
        ax_prob.plot(steps, probs, "-o", color=color, linewidth=2, markersize=5, label=short)

    for ax, ylab, title in [
        (ax_acc,  "accuracy",   "per-category accuracy vs. step"),
        (ax_prob, "prob_mean",  "per-category prob_mean vs. step"),
    ]:
        ax.set_xlabel("policy-update step")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=2)

    # --- bottom row: top-1 circuit head score (per circuit type) -------------
    for ax, (key, short, color) in zip(circuit_axes, CIRCUITS):
        scores = []
        heads  = []
        for s in steps:
            top = data[s]["circuits"].get(key, [])
            if top:
                L, H, sc = top[0][:3]
                scores.append(sc); heads.append((int(L), int(H)))
            else:
                scores.append(None); heads.append(None)
        ax.plot(steps, scores, "-o", color=color, linewidth=2, markersize=6)
        # annotate (L,H) at each point
        for x, y, lh in zip(steps, scores, heads):
            if y is None or lh is None: continue
            ax.annotate(f"L{lh[0]}H{lh[1]}", (x, y), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, ha="center", color=color)
        ax.set_xlabel("policy-update step")
        ax.set_ylabel("top-1 head score")
        ax.set_title(f"{short} circuit — best head vs. step")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label} — analysis across 200-step training ({run_tag})", fontsize=13)
    fig.tight_layout()
    out = BGRPO_ROOT / f"analysis_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  ({len(steps)} steps: {steps})")


def main() -> None:
    for run_tag, label in RUNS:
        plot_run(run_tag, label)


if __name__ == "__main__":
    main()
