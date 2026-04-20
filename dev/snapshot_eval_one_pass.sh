#!/bin/bash
# One-shot: snapshot the 3 training checkpoints, submit 6 eval jobs
# (greedy+beam7 for each of d=256/512/768), wait for them, then rewrite
# the snapshot history CSV that feeds the loss-vs-accuracy plot.
#
# Appends one row per model to ``dev/plots/snapshot_history.csv``.

set -euo pipefail

REPO="/resnick/groups/Hippo/jaeha/PolynomialDecomposition"
MONITOR_DIR="$REPO/data_storage/things_on_paper/monitor"   # gitignored — monitor artefacts live here
mkdir -p "$MONITOR_DIR"
cd "$REPO"
source "$REPO/.venv/bin/activate"

TS=$(date +%Y%m%d_%H%M%S)
HUMAN_TS=$(date +"%Y-%m-%d %H:%M:%S")
echo "=== $HUMAN_TS — pass start ==="

# CSV paths used both for rotation (below) and step 4 (append). Hoisted here
# so step 1a can consult the CSV before rotating.
CSV="$MONITOR_DIR/snapshot_history.csv"
ACSV="$MONITOR_DIR/analysis_history.csv"

# --- helper functions (hoisted so earlier steps can call them) ---
latest_train_iter_and_loss() {
    # "$@" = one or more training-job slurm files (ordered oldest-to-newest
    # for chained continuations). Returns "<global_iter> <valid_loss>".
    python3 "$REPO/dev/_parse_train_log.py" "$@"
}

# Read the training-job chain from the registry so chained-continuation logs
# are automatically picked up after the first job completes.
CHAIN_FILE="$MONITOR_DIR/train_chain.txt"
read_chain() {
    # $1 = model tag (d=256/512/768). Echoes space-separated slurm-*.out paths.
    local tag="$1"
    [[ -f "$CHAIN_FILE" ]] || return 0
    awk -v t="$tag" '!/^#/ && $1==t {for (i=2;i<=NF;i++) print "'"$REPO"'/slurm-" $i ".out"}' "$CHAIN_FILE" | xargs 2>/dev/null
}

# --- 1a. rotate old snapshots whose evaluations are terminal (done or dead) ---
# Delete a snapshot if:
#   (a) its (model, timestamp) already has a CSV row, OR
#   (b) it has no sidecar .jobs file (orphan from a crashed pass), OR
#   (c) its sidecar lists only non-running SLURM jobs — meaning the evals are
#       either completed-but-didn't-write-CSV or cancelled/failed/timed out.
# Keep the snapshot if any of its sidecar jobs is still PENDING/RUNNING.
# Never touches ``_best.pt`` or ``_snapshot_best.pt`` (pattern mismatch).
python3 - "$REPO/data_storage/things_on_paper/model" "$CSV" <<'PY' || true
import csv, glob, os, re, subprocess, sys
model_dir, csv_path = sys.argv[1], sys.argv[2]

evaluated = set()
if os.path.isfile(csv_path):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ts = row.get("timestamp", "")
            ts_c = ts.replace("-", "").replace(" ", "_").replace(":", "")
            evaluated.add((row.get("model", ""), ts_c))

RUNNING_STATES = {"PENDING", "RUNNING", "REQUEUED", "SUSPENDED",
                  "CONFIGURING", "COMPLETING", "RESIZING"}

def job_state(jid: str) -> str:
    """Best-effort single-job state via sacct; empty if unknown."""
    try:
        out = subprocess.check_output(
            ["sacct", "-j", jid, "-n", "-X", "-o", "State"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        return out.strip().split("\n")[0].strip().split()[0] if out.strip() else ""
    except Exception:
        return ""

JOB_KEYS = {"greedy", "beam", "analysis"}

def any_still_running(sidecar: str) -> bool:
    """True if any job in the sidecar is still PENDING/RUNNING."""
    try:
        lines = [l.strip().split() for l in open(sidecar) if l.strip() and not l.startswith("#")]
    except OSError:
        return False
    for kv in lines:
        if len(kv) < 2 or kv[0] not in JOB_KEYS:
            continue
        st = job_state(kv[1])
        if st in RUNNING_STATES or st == "":
            return True   # unknown state → be safe, keep
    return False

pat = re.compile(r"^d2_arch_(\d+)_l6_best_snapshot_(\d{8}_\d{6})\.pt$")
removed_done = removed_dead = removed_orphan = kept = 0
for f in sorted(glob.glob(os.path.join(model_dir, "d2_arch_*_l6_best_snapshot_*.pt"))):
    name = os.path.basename(f)
    m = pat.match(name)
    if not m:
        continue
    D, ts = m.group(1), m.group(2)
    sidecar = f[:-3] + ".jobs"

    if (f"d={D}", ts) in evaluated:
        os.remove(f)
        if os.path.exists(sidecar): os.remove(sidecar)
        print(f"  rotated (eval done): {name}")
        removed_done += 1
        continue

    if not os.path.exists(sidecar):
        os.remove(f)
        print(f"  rotated (orphan — no sidecar): {name}")
        removed_orphan += 1
        continue

    if any_still_running(sidecar):
        print(f"  keeping (eval pending): {name}")
        kept += 1
    else:
        os.remove(f); os.remove(sidecar)
        print(f"  rotated (eval dead — jobs terminal w/o CSV row): {name}")
        removed_dead += 1

print(f"  rotation summary: done={removed_done}, dead={removed_dead}, "
      f"orphan={removed_orphan}, kept_pending={kept}")
PY

# --- 1b. snapshot the current training best (new timestamp) ---
# Skip when the training iter hasn't advanced since the most recent sidecar —
# typically this means the model's chained job is still pending on afterok
# (no new training happening), so a fresh snapshot would be identical.
declare -A SNAP
SNAP_ITER_STEP_THRESHOLD=50
for D in 256 512 768; do
    SRC="$REPO/data_storage/things_on_paper/model/d2_arch_${D}_l6_best.pt"
    DST="$REPO/data_storage/things_on_paper/model/d2_arch_${D}_l6_best_snapshot_${TS}.pt"
    if [[ ! -s "$SRC" ]]; then
        echo "WARN d=$D: no best checkpoint at $SRC — skipping snapshot"
        continue
    fi

    # shellcheck disable=SC2046
    CUR_INFO=$(latest_train_iter_and_loss $(read_chain "d=$D") 2>/dev/null || echo "0 NaN")
    CUR_ITER=${CUR_INFO%% *}

    # Find the iter recorded in the most recent prior sidecar for this model.
    PREV_ITER=
    PREV_SIDECAR=$(ls -t "$REPO/data_storage/things_on_paper/model/d2_arch_${D}_l6_best_snapshot_"*.jobs 2>/dev/null | head -1)
    if [[ -n "$PREV_SIDECAR" ]]; then
        PREV_ITER=$(awk '$1=="iter"{print $2; exit}' "$PREV_SIDECAR")
    fi
    if [[ -n "$PREV_ITER" && "$PREV_ITER" =~ ^[0-9]+$ && "$CUR_ITER" =~ ^[0-9]+$ ]]; then
        DIFF=$(( CUR_ITER - PREV_ITER ))
        if (( DIFF < SNAP_ITER_STEP_THRESHOLD )); then
            echo "  d=$D: iter advanced only $DIFF (<$SNAP_ITER_STEP_THRESHOLD) since last snapshot — skipping (training likely pending)"
            continue
        fi
    fi

    cp "$SRC" "$DST"
    SNAP[$D]="d2_arch_${D}_l6_best_snapshot_${TS}"
    echo "  d=$D snapshot → $(basename "$DST")  (iter=$CUR_ITER)"
done

# --- 2. submit 9 jobs: greedy + beam eval + analysis combo for each model ---
declare -A GREEDY_JOB BEAM_JOB ANALYSIS_JOB ANALYSIS_OUT
BEAM_WIDTH=30
BEAM_MAX_TEST=30   # beam-30 is ~4x slower than beam-7; keep sample count low enough to fit a 15-min cycle
for D in 256 512 768; do
    [[ -z "${SNAP[$D]:-}" ]] && continue
    GREEDY_JOB[$D]=$(CKPT_TAG="${SNAP[$D]}" MAX_TEST=200 sbatch --parsable \
        Training/things_on_paper/eval/run_greedy_eval.sh \
        Training/things_on_paper/configs/d2_arch_${D}_l6.env test)
    BEAM_JOB[$D]=$(CKPT_TAG="${SNAP[$D]}" MAX_TEST="$BEAM_MAX_TEST" sbatch --parsable \
        Training/things_on_paper/eval/run_beam_eval.sh \
        Training/things_on_paper/configs/d2_arch_${D}_l6.env test "$BEAM_WIDTH")
    # Write analysis JSON into a per-snapshot dir so parallel passes don't overwrite each other.
    ANALYSIS_OUT[$D]="$MONITOR_DIR/analysis_${SNAP[$D]}"
    ANALYSIS_JOB[$D]=$(CKPT_TAG="${SNAP[$D]}" OUT_DIR="${ANALYSIS_OUT[$D]}" N_SAMPLES=200 LINE_IDX=0 \
        sbatch --parsable \
        Training/things_on_paper/analysis/run_analysis_combined.sh \
        Training/things_on_paper/configs/d2_arch_${D}_l6.env)
    # Sidecar .jobs file next to the snapshot. Rotation at the next tick
    # consults it to tell "eval still queued" (keep) from "eval dead" (delete).
    # We also record the training iteration and valid_loss at snapshot time so
    # post-hoc inspection of a snapshot doesn't need to cross-reference logs.
    SIDECAR="$REPO/data_storage/things_on_paper/model/${SNAP[$D]}.jobs"
    # shellcheck disable=SC2046
    SNAP_TRAIN_INFO=$(latest_train_iter_and_loss $(read_chain "d=$D") 2>/dev/null || echo "0 NaN")
    SNAP_ITER=${SNAP_TRAIN_INFO%% *}
    SNAP_VLOSS=${SNAP_TRAIN_INFO##* }
    {
        echo "# snapshot metadata"
        echo "timestamp $HUMAN_TS"
        echo "iter $SNAP_ITER"
        echo "valid_loss $SNAP_VLOSS"
        echo "greedy ${GREEDY_JOB[$D]}"
        echo "beam ${BEAM_JOB[$D]}"
        echo "analysis ${ANALYSIS_JOB[$D]}"
    } > "$SIDECAR"
    echo "  d=$D submitted: greedy=${GREEDY_JOB[$D]} beam${BEAM_WIDTH}=${BEAM_JOB[$D]} analysis=${ANALYSIS_JOB[$D]}"
done

# --- 3. wait for all 9 to finish (or timeout at 15 min) ---
DEADLINE=$(( $(date +%s) + 900 ))
while (( $(date +%s) < DEADLINE )); do
    RUNNING=0
    for D in 256 512 768; do
        for J in "${GREEDY_JOB[$D]:-}" "${BEAM_JOB[$D]:-}" "${ANALYSIS_JOB[$D]:-}"; do
            [[ -z "$J" ]] && continue
            STATE=$(sacct -j "$J" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')
            case "$STATE" in
                COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY) ;;
                *) RUNNING=$((RUNNING+1)) ;;
            esac
        done
    done
    (( RUNNING == 0 )) && break
    sleep 10
done

# --- 4. parse accuracies, append to CSV ---
if [[ ! -s "$CSV" ]]; then
    echo "timestamp,model,greedy_correct,greedy_total,greedy_acc,beam7_correct,beam7_total,beam7_acc,valid_loss,train_iter,beam_slurm_job" > "$CSV"
fi

extract_correct() {
    # $1 = slurm-<N>.out path. Extracts "Correct: X.0 out of Y.0" → integer pair "X Y".
    # %d coercion drops the ".0" float suffix without concatenating digits.
    grep -E "^Correct:" "$1" 2>/dev/null | tail -1 | awk '{printf "%d %d\n", $2, $5}'
}

extract_beam() {
    # $1 = slurm-<N>.out path, $2 = beam width. Extracts "Beam width $2: X out of Y"
    tail -c 3000 "$1" 2>/dev/null | tr '\r' '\n' \
        | grep -E "^Beam width $2:" | tail -1 | awk '{printf "%d %d\n", $4, $7}'
}

for D in 256 512 768; do
    [[ -z "${SNAP[$D]:-}" ]] && continue
    g="$REPO/slurm-${GREEDY_JOB[$D]:-X}.out"
    b="$REPO/slurm-${BEAM_JOB[$D]:-X}.out"
    greedy_pair=$(extract_correct "$g")
    # CSV legacy "beam7" columns track width 7 from the sweep so the
    # historic chart keeps comparing like with like.
    beam_pair=$(extract_beam "$b" 7)
    gc=$(echo "$greedy_pair" | awk '{print $1+0}')
    gt=$(echo "$greedy_pair" | awk '{print $2+0}')
    bc=$(echo "$beam_pair" | awk '{print $1+0}')
    bt=$(echo "$beam_pair" | awk '{print $2+0}')
    gacc=$(awk -v c="$gc" -v t="$gt" 'BEGIN{if(t>0) printf "%.3f", 100*c/t; else print "NaN"}')
    bacc=$(awk -v c="$bc" -v t="$bt" 'BEGIN{if(t>0) printf "%.3f", 100*c/t; else print "NaN"}')
    # shellcheck disable=SC2046 # intentional word-splitting of space-separated paths
    train_info=$(latest_train_iter_and_loss $(read_chain "d=$D"))
    t_iter=${train_info%% *}
    t_vloss=${train_info##* }
    echo "$HUMAN_TS,d=$D,$gc,$gt,$gacc,$bc,$bt,$bacc,$t_vloss,$t_iter,${BEAM_JOB[$D]}" >> "$CSV"
    echo "  d=$D results: greedy=$gc/$gt ($gacc%)  beam${BEAM_WIDTH}(@7)=$bc/$bt ($bacc%)  valid_loss=$t_vloss @ global_iter $t_iter"
done

# --- 4b. pull analyzer scores (confusion.json + circuits.json) into a parallel CSV ---
if [[ ! -s "$ACSV" ]]; then
    echo "timestamp,model,train_iter,sign_acc,op_acc,num_acc,var_acc,sign_prob,op_prob,num_prob,prev_top_score,prev_top_LH,within_top_score,within_top_LH,delim_top_score,delim_top_LH,analysis_slurm_job" > "$ACSV"
fi

extract_analysis_row() {
    # Arg 1 = confusion.json path, arg 2 = circuits.json path.
    # Emits one CSV-ready fragment "sign_acc,op_acc,...,delim_top_LH" or empty
    # string if either JSON is missing. Uses python for robust parsing.
    python3 - "$1" "$2" <<'PY' 2>/dev/null || true
import json, sys
try:
    conf = json.load(open(sys.argv[1]))
    cir = json.load(open(sys.argv[2]))
except Exception:
    sys.exit(0)
def pick(d, k, field, default=""):
    return f"{d[k][field]:.3f}" if k in d and field in d[k] else str(default)
def top(lst):
    if not lst:
        return ("", "")
    L, H, score = lst[0]
    return (f"{score:.3f}", f"L{L}H{H}")
row = ",".join([
    pick(conf, "SIGN", "acc"),
    pick(conf, "OPERATOR", "acc"),
    pick(conf, "NUMBER", "acc"),
    pick(conf, "VARIABLE", "acc"),
    pick(conf, "SIGN", "prob_mean"),
    pick(conf, "OPERATOR", "prob_mean"),
    pick(conf, "NUMBER", "prob_mean"),
    *top(cir.get("previous_token_top", [])),
    *top(cir.get("within_monomial_top", [])),
    *top(cir.get("delimiter_top", [])),
])
print(row)
PY
}

for D in 256 512 768; do
    [[ -z "${SNAP[$D]:-}" ]] && continue
    ADIR="${ANALYSIS_OUT[$D]:-}"
    [[ -z "$ADIR" || ! -f "$ADIR/confusion.json" || ! -f "$ADIR/circuits.json" ]] && {
        echo "  d=$D: analysis JSON missing — skipping analyzer CSV row"
        continue
    }
    # pull iter from the same train-log we already parsed above
    # shellcheck disable=SC2046
    t_info=$(latest_train_iter_and_loss $(read_chain "d=$D"))
    t_iter=${t_info%% *}
    row=$(extract_analysis_row "$ADIR/confusion.json" "$ADIR/circuits.json")
    if [[ -n "$row" ]]; then
        echo "$HUMAN_TS,d=$D,$t_iter,$row,${ANALYSIS_JOB[$D]}" >> "$ACSV"
        echo "  d=$D analyzer: $row"
    fi
done

# --- 4c. preserve best-ever beam7 snapshots as _snapshot_best.pt ---
# Runs after CSV append so "this pass" is already part of the aggregation.
python3 - "$CSV" "$HUMAN_TS" "$REPO" "${SNAP[256]:-}" "${SNAP[512]:-}" "${SNAP[768]:-}" <<'PY' || true
import csv, os, shutil, sys
csv_path, ts, repo, *snap_stems = sys.argv[1:]
snap_by_d = dict(zip(("256", "512", "768"), snap_stems))
best: dict[str, float] = {}      # model → max beam7_acc seen
this: dict[str, float] = {}      # model → beam7_acc at ts (this pass)
with open(csv_path) as f:
    for row in csv.DictReader(f):
        try:
            b = float(row["beam7_acc"])
        except (KeyError, ValueError):
            continue
        m = row["model"]
        best[m] = max(best.get(m, -1.0), b)
        if row["timestamp"] == ts:
            this[m] = b
model_dir = f"{repo}/data_storage/things_on_paper/model"
for d_key, b in this.items():
    D = d_key.split("=")[1]
    stem = snap_by_d.get(D, "")
    if not stem or b < best[d_key]:
        continue
    src = f"{model_dir}/{stem}.pt"
    dst = f"{model_dir}/d2_arch_{D}_l6_snapshot_best.pt"
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"  d={D}: new best-ever beam7 ({b:.2f}%) preserved → {os.path.basename(dst)}")
PY

# --- 5. regenerate the plots ---
python3 "$REPO/dev/plot_snapshot_history.py" || echo "plot regeneration failed (non-fatal)"
echo "=== pass complete ==="
