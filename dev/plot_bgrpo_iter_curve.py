"""Plot accuracy vs. policy-update step for the grpo / bgrpo / bgrpo_rank runs.

Reads `summary.json` files from
    data_storage/things_on_paper/BGRPO/runs/<run>/eval/<ckpt>/summary.json
and plots both greedy and beam@30 accuracy against the checkpoint step.

Checkpoint "final" is treated as == the largest numeric step (TRL saves both
at the end of training).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/resnick/groups/Hippo/jaeha/PolynomialDecomposition")
BGRPO_ROOT = REPO / "data_storage/things_on_paper/BGRPO/runs"
OUT = BGRPO_ROOT.parent / "bgrpo_iter_curve.png"

RUNS = [
    ("grpo",       "vanilla GRPO",       "#1f77b4", "o"),
    ("bgrpo",      "BGRPO (binary)",     "#2ca02c", "s"),
    ("bgrpo_rank", "BGRPO (rank-aware)", "#d62728", "^"),
]
# SFT init = d=256 snapshot_best, beam@30 on the same 60-test set.
# grpo/ckpt-10 sits at 26.667% and vanilla GRPO is flat → use it as the
# iter=0 anchor for all three curves.
SFT_BEAM30 = 26.667
SFT_GREEDY = 12.5  # grpo/ckpt-10 greedy


# All three runs trained for 25 policy-update steps; `checkpoint-final` is
# TRL's alias for the last step (ckpt-25 and ckpt-final are byte-identical).
FINAL_STEP = 25


def load_run(run: str) -> tuple[list[int], list[float], list[float]]:
    data: dict[int, tuple[float, float]] = {}
    for p in sorted((BGRPO_ROOT / run / "eval").glob("checkpoint-*/summary.json")):
        name = p.parent.name.removeprefix("checkpoint-")
        n = FINAL_STEP if name == "final" else int(name)
        summ = json.loads(p.read_text())
        greedy = summ.get("greedy", {}).get("acc")
        beam30 = summ.get("beam", {}).get("30", {}).get("acc")
        if greedy is None or beam30 is None:
            continue
        # Later writes (e.g. an explicit ckpt-25 after ckpt-final) overwrite.
        data[n] = (greedy, beam30)
    steps = sorted(data)
    greedies = [data[s][0] for s in steps]
    beams = [data[s][1] for s in steps]
    return steps, greedies, beams


fig, (ax_b, ax_g) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

for run, label, color, marker in RUNS:
    steps, greedy, beam = load_run(run)
    xs = [0] + steps
    ys_b = [SFT_BEAM30] + beam
    ys_g = [SFT_GREEDY] + greedy
    ax_b.plot(xs, ys_b, marker=marker, color=color, linewidth=2, markersize=7,
              label=label)
    ax_g.plot(xs, ys_g, marker=marker, color=color, linewidth=2, markersize=7,
              label=label)

for ax, title, ylab in [
    (ax_b, "beam@30 accuracy vs. policy-update step",  "beam@30 acc (%)"),
    (ax_g, "greedy accuracy vs. policy-update step",   "greedy acc (%)"),
]:
    ax.axhline(SFT_BEAM30 if ax is ax_b else SFT_GREEDY,
               color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="SFT init")
    ax.set_xlabel("policy-update step")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

fig.suptitle("(B)GRPO fine-tuning on d=256 SFT snapshot (60-problem test set)",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
