#!/bin/bash
# One-shot: once the salvage evals on bgrpo_*_v1_crashed checkpoints complete,
# persist the per-step (beam@30, greedy) numbers into a small CSV that the
# iter-curve plotter will keep consulting — then drop the ~1.1 GB of
# _v1_crashed checkpoint dirs since the graph has captured the useful signal.
#
# Idempotent: safe to run multiple times. Deletes v1_crashed dirs only when
# all 12 salvage eval jobs are terminal AND all 12 summary.json files exist.

set -uo pipefail

REPO="/resnick/groups/Hippo/jaeha/PolynomialDecomposition"
cd "$REPO"

BGRPO_ROOT="data_storage/things_on_paper/BGRPO"
CSV="$BGRPO_ROOT/v1_crashed_early_curve.csv"
SALVAGE_JOBS="63185099 63185100 63185101 63185102 63185103 63185104 63185105 63185106 63185107 63185108 63185109 63185110"

# 1. Refuse to proceed while any salvage eval is still non-terminal.
RUNNING_STATES="PENDING RUNNING REQUEUED SUSPENDED CONFIGURING COMPLETING RESIZING"
for j in $SALVAGE_JOBS; do
    state=$(sacct -j "$j" -n -X -o State 2>/dev/null | head -1 | awk '{print $1}')
    if echo "$RUNNING_STATES" | tr ' ' '\n' | grep -qx "$state"; then
        echo "salvage job $j still $state — skipping finalize"
        exit 0
    fi
done
echo "all 12 salvage evals terminal — persisting + cleaning up"

# 2. Dump per-ckpt numbers into CSV (idempotent overwrite).
python3 - <<'PY'
import csv, json
from pathlib import Path

REPO = Path("/resnick/groups/Hippo/jaeha/PolynomialDecomposition")
BGRPO_ROOT = REPO / "data_storage/things_on_paper/BGRPO"
OUT = BGRPO_ROOT / "v1_crashed_early_curve.csv"

rows = []
for run_dir in sorted(BGRPO_ROOT.glob("runs/bgrpo*_v1_crashed")):
    run_tag = run_dir.name.replace("_s200_d256_v1_crashed", "")
    for summary in sorted(run_dir.glob("eval/checkpoint-*/summary.json")):
        step_name = summary.parent.name.removeprefix("checkpoint-")
        try:
            step = int(step_name)
        except ValueError:
            continue
        d = json.loads(summary.read_text())
        greedy = d.get("greedy", {}).get("acc")
        beam30 = d.get("beam", {}).get("30", {}).get("acc")
        if greedy is None or beam30 is None:
            continue
        rows.append({"run_tag": run_tag, "step": step,
                     "greedy_acc": greedy, "beam30_acc": beam30})

rows.sort(key=lambda r: (r["run_tag"], r["step"]))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run_tag", "step", "greedy_acc", "beam30_acc"])
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows to {OUT}")
for r in rows:
    print(f"  {r['run_tag']:12s} step={r['step']:4d}  greedy={r['greedy_acc']:>6}%  beam30={r['beam30_acc']:>6}%")
PY

# 3. Regenerate s200 plot (plotter falls back to the CSV for missing steps).
source "$REPO/.venv/bin/activate"
python3 "$REPO/dev/plot_bgrpo_iter_curve_s200.py" || true

# 4. Delete the ~1.1 GB of crashed-run dirs now that numbers are persisted.
echo "removing v1_crashed ckpt dirs..."
rm -rf "$REPO/$BGRPO_ROOT/runs/bgrpo_rank_s200_d256_v1_crashed" \
       "$REPO/$BGRPO_ROOT/runs/bgrpo_s200_d256_v1_crashed"
echo "done. v1_crashed early-curve CSV kept at $CSV."
