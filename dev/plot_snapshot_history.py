"""
Regenerate the D2 training-progress plots from the accumulated
snapshot_history.csv produced by dev/snapshot_eval_one_pass.sh.

Three artifacts, all in dev/plots/:
  * d2_sft_progress.png     — epoch-aware training curves + beam-7 overlays
  * d2_loss_vs_acc.png      — valid_loss → beam-7 accuracy scatter
  * d2_beam_sweep.png       — beam-width → accuracy curves, one line per
                              snapshot per model (Fig-8 shape when fully
                              trained)
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
# Monitor artefacts live under data_storage/things_on_paper/ (gitignored)
# so the repo stays clean; point PLOTS there.
PLOTS = REPO / "data_storage" / "things_on_paper" / "monitor"
PLOTS.mkdir(parents=True, exist_ok=True)
CSV_PATH = PLOTS / "snapshot_history.csv"
ANALYSIS_CSV_PATH = PLOTS / "analysis_history.csv"

TRAIN_LOGS = {
    "d=256": REPO / "slurm-63156718.out",
    "d=512": REPO / "slurm-63157920.out",
    "d=768": REPO / "slurm-63157921.out",
}

# Epoch-aware iter timeline lives in _parse_train_log.py — import via path.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_parse_train_log", REPO / "dev" / "_parse_train_log.py"
)
_ptl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ptl)


BEAM_LINE = re.compile(r"^Beam width (\d+): (\d+) out of (\d+):")

# Paper Fig 8 numbers for 6-layer D2 baselines (SFT, before any BGRPO), at
# beam width 30. Listed here so every plot can overlay them as reference.
PAPER_SFT_BEAM30 = {"d=256": 26.1, "d=512": 29.5, "d=768": 32.1}


def _robust_float(s: str) -> float:
    """Tolerate trailing junk like '0.123.' from older CSV rows."""
    m = re.match(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        raise ValueError(f"cannot parse float from {s!r}")
    return float(m.group(0))


def read_history() -> dict[str, list[dict]]:
    history: dict[str, list[dict]] = defaultdict(list)
    if not CSV_PATH.exists():
        return history
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            try:
                row["beam7_acc"] = _robust_float(row["beam7_acc"])
                row["greedy_acc"] = _robust_float(row["greedy_acc"])
                row["valid_loss"] = _robust_float(row["valid_loss"])
                row["train_iter"] = int(row["train_iter"])
            except (ValueError, KeyError):
                continue
            row["beam_slurm_job"] = row.get("beam_slurm_job", "")
            history[row["model"]].append(row)
    return history


def plot_progress(history: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, ("d=256", "d=512", "d=768")):
        timeline = _ptl.iter_timeline(TRAIN_LOGS[name])
        iters = [g for g, _, _ in timeline]
        tr = [t for _, t, _ in timeline]
        vl = [v for _, _, v in timeline]

        if iters:
            ax.plot(iters, tr, color="#1f77b4", alpha=0.5, label="train loss", linewidth=0.8)
            ax.plot(iters, vl, color="#ff7f0e", alpha=0.85, label="valid loss", linewidth=1.4)
        ax.set_title(f"{name} (l=6, block=850)")
        ax.set_xlabel("global iteration (epoch-adjusted)")
        ax.set_ylabel("loss  (log scale)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        if vl:
            y_lo = max(0.05, min(vl) * 0.7)
            y_hi = max(vl + tr) * 1.2
            ax.set_ylim(y_lo, y_hi)

        ax2 = ax.twinx()
        ax2.set_ylabel("beam-7 acc (%)", color="#2ca02c")
        ax2.tick_params(axis="y", colors="#2ca02c")
        rows = history.get(name, [])
        if rows:
            xs = [r["train_iter"] for r in rows]
            ys = [r["beam7_acc"] for r in rows]
            ax2.plot(xs, ys, color="#2ca02c", linewidth=1.0, alpha=0.4)
            ax2.scatter(xs, ys, s=70, color="#2ca02c", edgecolor="white",
                        linewidth=1.2, zorder=5)
            for x, y in zip(xs, ys):
                ax2.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                             xytext=(6, 4), fontsize=8, color="#2ca02c", fontweight="bold")
        # Paper reference — dashed horizontal at the Fig-8 SFT beam-30 value.
        paper_ref = PAPER_SFT_BEAM30.get(name)
        if paper_ref is not None:
            ax2.axhline(paper_ref, linestyle="--", color="#7f7f7f", linewidth=1.0, alpha=0.7)
            ax2.annotate(
                f"paper SFT @beam30: {paper_ref:.1f}%",
                xy=(0.98, paper_ref), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7.5, color="#7f7f7f",
            )

        max_acc = max((r["beam7_acc"] for r in rows), default=10)
        ax2.set_ylim(0, max(paper_ref * 1.15 if paper_ref else 15, max_acc * 1.3, 15))

        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=9)

    n_snaps = sum(len(v) for v in history.values())
    fig.suptitle(
        f"D2 SFT progress — log-loss curves + beam-7 eval overlays  "
        f"(epoch-adjusted x-axis; {n_snaps} snapshot rows)",
        y=1.02,
    )
    fig.tight_layout()
    out = PLOTS / "d2_sft_progress.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_loss_vs_acc(history: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"d=256": "#1f77b4", "d=512": "#ff7f0e", "d=768": "#2ca02c"}
    for model, rows in history.items():
        if not rows:
            continue
        xs = [r["valid_loss"] for r in rows]
        ys = [r["beam7_acc"] for r in rows]
        ax.plot(xs, ys, "o-", color=colors.get(model, "gray"),
                label=model, markersize=8, alpha=0.8)
        for x, y, r in zip(xs, ys, rows):
            ax.annotate(f"iter {r['train_iter']}", (x, y),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color=colors.get(model, "gray"))
    # Paper reference — faint dashed line per model at Fig-8 SFT beam-30 acc.
    for name, paper_ref in PAPER_SFT_BEAM30.items():
        if name in history:
            ax.axhline(paper_ref, linestyle="--", color=colors.get(name, "gray"),
                       linewidth=0.8, alpha=0.5)
            ax.text(0.02, paper_ref + 0.3, f"paper {name} SFT @beam30: {paper_ref:.1f}%",
                    transform=ax.get_yaxis_transform(),
                    fontsize=7.5, color=colors.get(name, "gray"), alpha=0.9)

    ax.set_xlabel("validation loss")
    ax.set_ylabel("beam-7 accuracy (%)")
    ax.set_title("valid loss → beam-7 accuracy  (paper Fig 8 refs dashed)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = PLOTS / "d2_loss_vs_acc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def parse_beam_sweep(slurm_path: Path) -> list[tuple[int, int, int]]:
    """From a beam-eval slurm.out, return [(width, correct, total), ...] sorted."""
    if not slurm_path.exists():
        return []
    text = slurm_path.read_text(errors="ignore").replace("\r", "\n")
    by_width: dict[int, tuple[int, int]] = {}
    for line in text.splitlines():
        m = BEAM_LINE.match(line)
        if m:
            w, c, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
            # Keep the LATEST "Beam width W:" line per width — the eval prints
            # running stats and the last one is the final over all samples.
            by_width[w] = (c, t)
    return [(w, by_width[w][0], by_width[w][1]) for w in sorted(by_width)]


def plot_beam_sweep(history: dict[str, list[dict]]) -> None:
    """One subplot per model; x = beam width, y = accuracy; one colored line
    per snapshot (alpha scales with recency)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, name in zip(axes, ("d=256", "d=512", "d=768")):
        rows = history.get(name, [])
        if not rows:
            ax.set_title(f"{name} — no data")
            continue
        # Only snapshots with a beam-sweep slurm file we can parse.
        plotted = 0
        n = len(rows)
        for idx, row in enumerate(rows):
            job = row.get("beam_slurm_job", "")
            if not job:
                continue
            slurm_file = REPO / f"slurm-{job}.out"
            sweep = parse_beam_sweep(slurm_file)
            if not sweep:
                continue
            widths = [w for w, _, _ in sweep]
            accs = [100 * c / t if t else 0 for _, c, t in sweep]
            # Alpha: older snapshots faded, latest opaque.
            alpha = 0.25 + 0.75 * (idx + 1) / n
            ax.plot(widths, accs, marker="o", markersize=4,
                    linewidth=1.2, alpha=alpha,
                    label=f"iter {row['train_iter']}")
            plotted += 1
        # Paper reference — dashed horizontal at SFT beam-30.
        paper_ref = PAPER_SFT_BEAM30.get(name)
        if paper_ref is not None:
            ax.axhline(paper_ref, linestyle="--", color="#7f7f7f", linewidth=1.0, alpha=0.7)
            ax.text(0.98, paper_ref + 0.3, f"paper SFT @30: {paper_ref:.1f}%",
                    transform=ax.get_yaxis_transform(),
                    ha="right", fontsize=7.5, color="#7f7f7f")

        ax.set_title(f"{name}  ({plotted} sweeps)")
        ax.set_xlabel("beam width")
        ax.set_ylabel("accuracy (%)")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Beam-width sweep per snapshot (newer = darker)", y=1.02)
    fig.tight_layout()
    out = PLOTS / "d2_beam_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def read_analysis_history() -> dict[str, list[dict]]:
    """Load the per-snapshot analyzer scores CSV (may not exist yet)."""
    out: dict[str, list[dict]] = defaultdict(list)
    if not ANALYSIS_CSV_PATH.exists():
        return out
    with ANALYSIS_CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            try:
                row["train_iter"] = int(row["train_iter"])
                for k in ("sign_acc", "op_acc", "num_acc", "var_acc",
                         "sign_prob", "op_prob", "num_prob",
                         "prev_top_score", "within_top_score", "delim_top_score"):
                    row[k] = _robust_float(row[k]) if row.get(k) else float("nan")
            except (ValueError, KeyError):
                continue
            out[row["model"]].append(row)
    return out


# Paper Table 4 — 6-layer SFT accuracy numbers for overlay on the analyzer plot.
PAPER_CONFUSION_ACC = {
    "SIGN":     {"d=256": 0.522, "d=512": 0.523, "d=768": 0.521},
    "OPERATOR": {"d=256": 0.943, "d=512": 0.941, "d=768": 0.942},
    "NUMBER":   {"d=256": 0.911, "d=512": 0.905, "d=768": 0.903},
}


def plot_analysis_progress(analysis: dict[str, list[dict]]) -> None:
    """Two rows × three cols: top row = confusion-accuracy-by-category per
    model; bottom row = top circuit score for prev-token / within-monomial /
    delimiter heads per model. Dashed horizontals are paper Table-4 values
    where known."""
    if not any(analysis.values()):
        print(f"skip analysis plot — {ANALYSIS_CSV_PATH.name} empty")
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    colors = {"d=256": "#1f77b4", "d=512": "#ff7f0e", "d=768": "#2ca02c"}

    # --- top row: per-category greedy accuracy (SIGN / OPERATOR / NUMBER) ---
    cats = [("sign_acc", "SIGN"), ("op_acc", "OPERATOR"), ("num_acc", "NUMBER")]
    for ax, (key, cat_name) in zip(axes[0], cats):
        for model, rows in analysis.items():
            if not rows:
                continue
            xs = [r["train_iter"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, "o-", color=colors.get(model, "gray"),
                    label=model, markersize=5, alpha=0.85)
            paper_ref = PAPER_CONFUSION_ACC.get(cat_name, {}).get(model)
            if paper_ref is not None:
                ax.axhline(paper_ref, linestyle="--",
                           color=colors.get(model, "gray"),
                           linewidth=0.8, alpha=0.5)
        ax.set_title(f"{cat_name}  greedy acc  (paper Table 4 dashed)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if ax is axes[0, 0]:
            ax.legend(fontsize=8)

    # --- bottom row: top circuit score per model ---
    circuits = [
        ("prev_top_score", "previous-token head (L0 paper circuit)"),
        ("within_top_score", "within-monomial head"),
        ("delim_top_score", "delimiter head (L1 paper circuit)"),
    ]
    for ax, (key, title) in zip(axes[1], circuits):
        for model, rows in analysis.items():
            if not rows:
                continue
            xs = [r["train_iter"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, "o-", color=colors.get(model, "gray"),
                    label=model, markersize=5, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("global iteration")
        ax.set_ylabel("top head score")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    n_rows = sum(len(v) for v in analysis.values())
    fig.suptitle(f"D2 analyzer-score progress  ({n_rows} snapshot rows)", y=1.01)
    fig.tight_layout()
    out = PLOTS / "d2_analysis_progress.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    history = read_history()
    plot_progress(history)
    plot_loss_vs_acc(history)
    plot_beam_sweep(history)
    analysis = read_analysis_history()
    plot_analysis_progress(analysis)


if __name__ == "__main__":
    main()
