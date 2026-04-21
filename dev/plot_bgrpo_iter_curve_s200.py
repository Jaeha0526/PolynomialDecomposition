"""Plot the 200-step (B)GRPO learning curves.

Reads `summary.json` from
    data_storage/things_on_paper/BGRPO/runs/{grpo,bgrpo,bgrpo_rank}_s200_d256/eval/<ckpt>/summary.json

For bgrpo and bgrpo_rank the first attempts (v1) crashed around outer 75-80
due to a sympy-None bug; their early-step evals were captured into
`data_storage/things_on_paper/BGRPO/v1_crashed_early_curve.csv` before the
crashed-run dirs were deleted. We fall back to that CSV whenever a step is
missing from the live eval dir — so the graph stays faithful even after
the crashed-run weights are cleaned up.

checkpoint-final maps to step 200 (TRL byte-equivalent alias).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/resnick/groups/Hippo/jaeha/PolynomialDecomposition")
BGRPO_ROOT = REPO / "data_storage/things_on_paper/BGRPO"
OUT = BGRPO_ROOT / "bgrpo_iter_curve_s200.png"
V1_CSV = BGRPO_ROOT / "v1_crashed_early_curve.csv"

RUNS = [
    ("grpo_s200_d256",       "vanilla GRPO",       "#1f77b4", "o", None),
    ("bgrpo_s200_d256",      "BGRPO (binary)",     "#2ca02c", "s", "bgrpo"),
    ("bgrpo_rank_s200_d256", "BGRPO (rank-aware)", "#d62728", "^", "bgrpo_rank"),
]
# SFT anchor = d=256 snapshot_best on the 60-problem test set
# (= grpo_s200/ckpt-10's beam@30 since vanilla GRPO is flat there).
SFT_BEAM30 = 26.667
SFT_GREEDY = 12.5

FINAL_STEP = 200


def load_v1_csv() -> dict[tuple[str, int], tuple[float, float]]:
    """{(v1_run_tag, step): (greedy, beam30)} sourced from the persisted CSV."""
    if not V1_CSV.exists():
        return {}
    out: dict[tuple[str, int], tuple[float, float]] = {}
    with open(V1_CSV) as f:
        for row in csv.DictReader(f):
            try:
                out[(row["run_tag"], int(row["step"]))] = (
                    float(row["greedy_acc"]), float(row["beam30_acc"]))
            except (KeyError, ValueError):
                continue
    return out


def load_run(run_dir_name: str, v1_tag: str | None,
             v1_data: dict[tuple[str, int], tuple[float, float]]
             ) -> tuple[list[int], list[float], list[float]]:
    data: dict[int, tuple[float, float]] = {}
    run_dir = BGRPO_ROOT / "runs" / run_dir_name / "eval"
    if run_dir.exists():
        for p in sorted(run_dir.glob("checkpoint-*/summary.json")):
            name = p.parent.name.removeprefix("checkpoint-")
            n = FINAL_STEP if name == "final" else int(name)
            summ = json.loads(p.read_text())
            greedy = summ.get("greedy", {}).get("acc")
            beam30 = summ.get("beam", {}).get("30", {}).get("acc")
            if greedy is None or beam30 is None:
                continue
            data[n] = (greedy, beam30)
    # Fall back to v1 CSV for any missing early-step points.
    if v1_tag is not None:
        for (tag, step), pair in v1_data.items():
            if tag == v1_tag and step not in data:
                data[step] = pair
    steps = sorted(data)
    return steps, [data[s][0] for s in steps], [data[s][1] for s in steps]


def main() -> None:
    v1_data = load_v1_csv()
    fig, (ax_b, ax_g) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for run_dir, label, color, marker, v1_tag in RUNS:
        steps, greedy, beam = load_run(run_dir, v1_tag, v1_data)
        xs = [0] + steps
        ax_b.plot(xs, [SFT_BEAM30] + beam, marker=marker, color=color,
                  linewidth=2, markersize=6, label=label)
        ax_g.plot(xs, [SFT_GREEDY] + greedy, marker=marker, color=color,
                  linewidth=2, markersize=6, label=label)

    for ax, title, ylab, sft in [
        (ax_b, "beam@30 vs. policy-update step",  "beam@30 acc (%)", SFT_BEAM30),
        (ax_g, "greedy vs. policy-update step",   "greedy acc (%)",  SFT_GREEDY),
    ]:
        ax.axhline(sft, color="gray", linestyle="--", linewidth=1, alpha=0.5,
                   label="SFT init")
        ax.set_xlabel("policy-update step")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("(B)GRPO 200-step fine-tuning on d=256 SFT snapshot "
                 "(60-problem test set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
