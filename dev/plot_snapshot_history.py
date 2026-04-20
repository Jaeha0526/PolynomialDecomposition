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

# Training-log chain registry — one line per model, space-separated slurm IDs
# ordered oldest → newest. Chained continuations (afterok dependencies) get
# appended here by snapshot_eval_one_pass.sh; the plotter concatenates them.
CHAIN_FILE = PLOTS / "train_chain.txt"


def _read_chain() -> dict[str, list[Path]]:
    """Return {model_tag: [slurm-<id>.out, ...]} from the chain registry.

    Falls back to the hard-coded original three D2 jobs if the registry is
    missing (e.g. early in a fresh run). Any model not in the registry (new
    sibling branches, etc.) surfaces once rows appear in the CSV."""
    default = {
        "d=256": [REPO / "slurm-63156718.out"],
        "d=512": [REPO / "slurm-63157920.out"],
        "d=768": [REPO / "slurm-63157921.out"],
    }
    if not CHAIN_FILE.exists():
        return default
    out: dict[str, list[Path]] = {}
    for line in CHAIN_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        model, jobs = parts[0], parts[1:]
        out[model] = [REPO / f"slurm-{j}.out" for j in jobs]
    for k, v in default.items():
        out.setdefault(k, v)
    return out


TRAIN_LOGS = _read_chain()

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
# d=256b is our sibling-branch stable relaunch from the 35.7% snapshot;
# reuse the d=256 paper number since it's the same architecture.
PAPER_SFT_BEAM30 = {
    "d=256":   26.1, "d=512":   29.5, "d=768":   32.1,
    "d=256b":  26.1, "d=512r2": 29.5, "d=768r2": 32.1,
}

# Rounds: round 1 = original fresh runs; round 2 = stable-resume /
# from-scratch restarts. Plots for each round are emitted into separate files
# so they can be tracked independently.
ROUND_MODELS: dict[int, tuple[str, ...]] = {
    # Round 1 models d=256 / d=512 are STOPPED (no new evals) but their
    # historical CSV rows still plot — just the trace won't advance.
    1: ("d=256", "d=512", "d=768"),
    2: ("d=256b", "d=512r2", "d=768r2"),
}
MODEL_ORDER: tuple[str, ...] = ROUND_MODELS[1] + ROUND_MODELS[2]
MODEL_COLORS = {
    "d=256":   "#1f77b4",
    "d=512":   "#ff7f0e",
    "d=768":   "#2ca02c",
    "d=256b":  "#17becf",
    "d=512r2": "#d62728",
    "d=768r2": "#9467bd",
}


def _round_of(model: str) -> int | None:
    for r, ms in ROUND_MODELS.items():
        if model in ms:
            return r
    return None


def _models_for_round(history: dict, rnd: int) -> list[str]:
    """Return this round's models that have at least one CSV row (or a
    registered training chain), in ROUND_MODELS order."""
    return [m for m in ROUND_MODELS[rnd] if m in history or m in TRAIN_LOGS]


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


def plot_progress(history: dict[str, list[dict]], rnd: int) -> None:
    models = _models_for_round(history, rnd) or list(ROUND_MODELS[rnd])
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5),
                             squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, models):
        timeline = _ptl.iter_timeline(*TRAIN_LOGS[name])
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

    n_snaps = sum(len(history.get(m, [])) for m in models)
    fig.suptitle(
        f"D2 SFT progress — round {rnd}  "
        f"(epoch-adjusted x-axis; {n_snaps} snapshot rows)",
        y=1.02,
    )
    fig.tight_layout()
    out = PLOTS / f"d2_sft_progress_r{rnd}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_loss_vs_acc(history: dict[str, list[dict]], rnd: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = MODEL_COLORS
    subset = {m: history[m] for m in ROUND_MODELS[rnd] if m in history}
    for model, rows in subset.items():
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
    for name, paper_ref in PAPER_SFT_BEAM30.items():
        if name in subset:
            ax.axhline(paper_ref, linestyle="--", color=colors.get(name, "gray"),
                       linewidth=0.8, alpha=0.5)
            ax.text(0.02, paper_ref + 0.3, f"paper {name} SFT @beam30: {paper_ref:.1f}%",
                    transform=ax.get_yaxis_transform(),
                    fontsize=7.5, color=colors.get(name, "gray"), alpha=0.9)

    ax.set_xlabel("validation loss")
    ax.set_ylabel("beam-7 accuracy (%)")
    ax.set_title(f"valid loss → beam-7 accuracy  (round {rnd}; paper Fig 8 refs dashed)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    if subset:
        ax.legend()
    fig.tight_layout()
    out = PLOTS / f"d2_loss_vs_acc_r{rnd}.png"
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


def plot_beam_sweep(history: dict[str, list[dict]], rnd: int) -> None:
    """One subplot per model; x = beam width, y = accuracy. We highlight only
    (a) the top-3 sweeps by accuracy at beam width 30 and (b) the most-recent
    sweep. All other sweeps are rendered faintly as context. Legend lists
    only the 4 highlighted entries (fewer if top-3 overlaps with most-recent
    or fewer than 3 sweeps exist)."""
    models = _models_for_round(history, rnd) or list(ROUND_MODELS[rnd])
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, models):
        rows = history.get(name, [])
        if not rows:
            ax.set_title(f"{name} — no data")
            continue
        sweeps: list[tuple[int, dict, list[int], list[float], float]] = []
        for idx, row in enumerate(rows):
            job = row.get("beam_slurm_job", "")
            if not job:
                continue
            sweep = parse_beam_sweep(REPO / f"slurm-{job}.out")
            if not sweep:
                continue
            widths = [w for w, _, _ in sweep]
            accs = [100 * c / t if t else 0 for _, c, t in sweep]
            acc30 = next((a for w, a in zip(widths, accs) if w == 30), accs[-1])
            sweeps.append((idx, row, widths, accs, acc30))

        if not sweeps:
            ax.set_title(f"{name} — no data")
            continue

        # Pick highlights: top-3 by beam-30 acc, plus the most-recent by idx.
        top3 = sorted(sweeps, key=lambda s: s[4], reverse=True)[:3]
        most_recent = max(sweeps, key=lambda s: s[0])
        highlight_idxs = {s[0] for s in top3} | {most_recent[0]}

        highlight_colors = {
            top3[i][0]: c for i, c in zip(
                range(len(top3)), ("#d62728", "#ff7f0e", "#2ca02c")
            )
        }
        most_recent_color = "#1f77b4"

        # Background — faded, no legend.
        for idx, _row, widths, accs, _ in sweeps:
            if idx in highlight_idxs:
                continue
            ax.plot(widths, accs, marker="o", markersize=3,
                    linewidth=0.9, alpha=0.15, color="#7f7f7f")

        # Highlights — thick, opaque, legend-visible. Draw most-recent last so
        # it sits on top. If most-recent is also in top-3, keep the top-3 color
        # but tag the label with "(most recent)".
        drawn: set[int] = set()
        for idx, row, widths, accs, acc30 in sorted(
            top3, key=lambda s: s[4], reverse=True
        ):
            color = highlight_colors[idx]
            label = f"iter {row['train_iter']}  (top @30: {acc30:.1f}%)"
            if idx == most_recent[0]:
                label += "  — most recent"
            ax.plot(widths, accs, marker="o", markersize=5,
                    linewidth=1.8, alpha=0.95, color=color, label=label)
            drawn.add(idx)
        if most_recent[0] not in drawn:
            idx, row, widths, accs, acc30 = most_recent
            ax.plot(widths, accs, marker="o", markersize=5,
                    linewidth=1.8, alpha=0.95, color=most_recent_color,
                    label=f"iter {row['train_iter']}  — most recent (@30: {acc30:.1f}%)")

        paper_ref = PAPER_SFT_BEAM30.get(name)
        if paper_ref is not None:
            ax.axhline(paper_ref, linestyle="--", color="#7f7f7f",
                       linewidth=1.0, alpha=0.7)
            ax.text(0.98, paper_ref + 0.3, f"paper SFT @30: {paper_ref:.1f}%",
                    transform=ax.get_yaxis_transform(),
                    ha="right", fontsize=7.5, color="#7f7f7f")

        ax.set_title(f"{name}  ({len(sweeps)} sweeps)")
        ax.set_xlabel("beam width")
        ax.set_ylabel("accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"Beam-width sweep — round {rnd}  (top-3 by acc@30 + most recent "
        "highlighted; background sweeps faded)", y=1.02,
    )
    fig.tight_layout()
    out = PLOTS / f"d2_beam_sweep_r{rnd}.png"
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
    "SIGN":     {"d=256": 0.522, "d=512": 0.523, "d=768": 0.521,
                 "d=256b": 0.522, "d=512r2": 0.523, "d=768r2": 0.521},
    "OPERATOR": {"d=256": 0.943, "d=512": 0.941, "d=768": 0.942,
                 "d=256b": 0.943, "d=512r2": 0.941, "d=768r2": 0.942},
    "NUMBER":   {"d=256": 0.911, "d=512": 0.905, "d=768": 0.903,
                 "d=256b": 0.911, "d=512r2": 0.905, "d=768r2": 0.903},
}


def plot_analysis_progress(analysis: dict[str, list[dict]], rnd: int) -> None:
    """Two rows × three cols: top row = confusion-accuracy-by-category per
    model; bottom row = top circuit score for prev-token / within-monomial /
    delimiter heads per model. Dashed horizontals are paper Table-4 values
    where known."""
    subset = {m: analysis[m] for m in ROUND_MODELS[rnd] if m in analysis}
    if not any(subset.values()):
        print(f"skip analysis plot round {rnd} — no rows for {ROUND_MODELS[rnd]}")
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    colors = MODEL_COLORS

    # --- top row: per-category greedy accuracy (SIGN / OPERATOR / NUMBER) ---
    cats = [("sign_acc", "SIGN"), ("op_acc", "OPERATOR"), ("num_acc", "NUMBER")]
    for ax, (key, cat_name) in zip(axes[0], cats):
        for model, rows in subset.items():
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
        for model, rows in subset.items():
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

    n_rows = sum(len(v) for v in subset.values())
    fig.suptitle(f"D2 analyzer-score progress — round {rnd}  ({n_rows} snapshot rows)", y=1.01)
    fig.tight_layout()
    out = PLOTS / f"d2_analysis_progress_r{rnd}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    history = read_history()
    analysis = read_analysis_history()
    for rnd in sorted(ROUND_MODELS):
        plot_progress(history, rnd)
        plot_loss_vs_acc(history, rnd)
        plot_beam_sweep(history, rnd)
        plot_analysis_progress(analysis, rnd)


if __name__ == "__main__":
    main()
