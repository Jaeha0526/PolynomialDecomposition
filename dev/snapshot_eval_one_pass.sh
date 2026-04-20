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

# --- 0. back-fill CSV rows for sidecars whose eval jobs finished between passes ---
# If the previous pass's 15-min wait expired before the beam eval produced its
# final "Beam width 7: X out of Y:" line, the CSV row was written with zeros.
# The jobs keep running on SLURM; by now they may be done, so re-parse their
# slurm files and update / insert a proper row. Runs BEFORE rotation so the
# next step's "CSV row exists" check sees the real numbers.
python3 - "$REPO/data_storage/things_on_paper/model" "$REPO" "$CSV" "$ACSV" <<'PY' || true
import csv, glob, os, re, subprocess, sys
model_dir, repo, csv_path, acsv_path = sys.argv[1:]

RUNNING_STATES = {"PENDING", "RUNNING", "REQUEUED", "SUSPENDED",
                  "CONFIGURING", "COMPLETING", "RESIZING"}

def job_state(jid: str) -> str:
    try:
        out = subprocess.check_output(
            ["sacct", "-j", jid, "-n", "-X", "-o", "State"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        return out.strip().split("\n")[0].strip().split()[0] if out.strip() else ""
    except Exception:
        return ""

def read_sidecar(path: str) -> dict:
    out: dict[str, str] = {}
    try:
        for l in open(path):
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            kv = l.split(None, 1)
            if len(kv) == 2:
                out[kv[0]] = kv[1]
    except OSError:
        pass
    return out

def extract_greedy(slurm_path: str) -> tuple[int, int]:
    if not os.path.isfile(slurm_path):
        return 0, 0
    last = None
    for line in open(slurm_path, errors="ignore"):
        if line.startswith("Correct:"):
            last = line
    if not last:
        return 0, 0
    m = re.search(r"Correct:\s*([\d.]+)\s+out of\s+([\d.]+)", last)
    return (int(float(m.group(1))), int(float(m.group(2)))) if m else (0, 0)

def extract_beam_all(slurm_path: str) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    if not os.path.isfile(slurm_path):
        return out
    pat = re.compile(r"^Beam width (\d+):\s+(\d+) out of (\d+):")
    for line in open(slurm_path, errors="ignore"):
        m = pat.match(line)
        if m:
            out[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    return out

def extract_analysis(out_dir: str) -> dict:
    import json
    conf_p = os.path.join(out_dir, "confusion.json")
    cir_p = os.path.join(out_dir, "circuits.json")
    if not (os.path.isfile(conf_p) and os.path.isfile(cir_p)):
        return {}
    try:
        conf = json.load(open(conf_p))
        cir = json.load(open(cir_p))
    except Exception:
        return {}
    def pick(d, k, f): return f"{d[k][f]:.3f}" if k in d and f in d[k] else ""
    def top(lst):
        if not lst: return ("", "")
        L, H, s = lst[0]; return (f"{s:.3f}", f"L{L}H{H}")
    row = {
        "sign_acc": pick(conf, "SIGN", "acc"),
        "op_acc": pick(conf, "OPERATOR", "acc"),
        "num_acc": pick(conf, "NUMBER", "acc"),
        "var_acc": pick(conf, "VARIABLE", "acc"),
        "sign_prob": pick(conf, "SIGN", "prob_mean"),
        "op_prob": pick(conf, "OPERATOR", "prob_mean"),
        "num_prob": pick(conf, "NUMBER", "prob_mean"),
    }
    row["prev_top_score"], row["prev_top_LH"] = top(cir.get("previous_token_top", []))
    row["within_top_score"], row["within_top_LH"] = top(cir.get("within_monomial_top", []))
    row["delim_top_score"], row["delim_top_LH"] = top(cir.get("delimiter_top", []))
    return row

def read_csv_rows(path: str) -> tuple[list[str], list[dict]]:
    if not os.path.isfile(path):
        return [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)

def write_csv_rows(path: str, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def upsert(rows: list[dict], key: tuple[str, str], new: dict) -> tuple[list[dict], bool]:
    """Replace existing (model, timestamp) row OR append. Return (rows, updated_flag)."""
    for i, r in enumerate(rows):
        if (r.get("model", ""), r.get("timestamp", "")) == key:
            # only rewrite if new row brings real data (beam7_total > 0) and old had 0s
            old_bt = int(float(r.get("beam7_total") or 0) or 0)
            new_bt = int(float(new.get("beam7_total") or 0) or 0)
            if new_bt > old_bt:
                rows[i] = new
                return rows, True
            return rows, False
    rows.append(new); return rows, True

pat = re.compile(r"^d2_arch_([0-9a-z]+)_l6_best_snapshot_(\d{8}_\d{6})\.jobs$")
csv_fields, csv_rows = read_csv_rows(csv_path)
acsv_fields, acsv_rows = read_csv_rows(acsv_path)
if not csv_fields:
    csv_fields = ["timestamp", "model", "greedy_correct", "greedy_total", "greedy_acc",
                  "beam7_correct", "beam7_total", "beam7_acc", "valid_loss",
                  "train_iter", "beam_slurm_job"]
if not acsv_fields:
    acsv_fields = ["timestamp", "model", "train_iter",
                   "sign_acc", "op_acc", "num_acc", "var_acc",
                   "sign_prob", "op_prob", "num_prob",
                   "prev_top_score", "prev_top_LH",
                   "within_top_score", "within_top_LH",
                   "delim_top_score", "delim_top_LH", "analysis_slurm_job"]

backfilled = 0
for sidecar in sorted(glob.glob(os.path.join(model_dir, "d2_arch_*_l6_best_snapshot_*.jobs"))):
    name = os.path.basename(sidecar)
    m = pat.match(name)
    if not m:
        continue
    D, ts_c = m.group(1), m.group(2)
    meta = read_sidecar(sidecar)
    gid = meta.get("greedy", ""); bid = meta.get("beam", ""); aid = meta.get("analysis", "")
    # need beam + greedy to build main-CSV row; analysis separate
    if not (gid and bid):
        continue
    # only back-fill if both jobs are in terminal state
    states = [job_state(j) for j in (gid, bid)]
    if any(s in RUNNING_STATES or s == "" for s in states):
        continue
    hts = meta.get("timestamp", f"{ts_c[:4]}-{ts_c[4:6]}-{ts_c[6:8]} {ts_c[9:11]}:{ts_c[11:13]}:{ts_c[13:15]}")
    model_key = f"d={D}"
    # parse slurm files
    gslurm = os.path.join(repo, f"slurm-{gid}.out")
    bslurm = os.path.join(repo, f"slurm-{bid}.out")
    gc, gt = extract_greedy(gslurm)
    beam_by_width = extract_beam_all(bslurm)
    bc, bt = beam_by_width.get(7, (0, 0))
    gacc = f"{100*gc/gt:.3f}" if gt else "NaN"
    bacc = f"{100*bc/bt:.3f}" if bt else "NaN"
    new_row = {
        "timestamp": hts, "model": model_key,
        "greedy_correct": str(gc), "greedy_total": str(gt), "greedy_acc": gacc,
        "beam7_correct": str(bc), "beam7_total": str(bt), "beam7_acc": bacc,
        "valid_loss": meta.get("valid_loss", "NaN"),
        "train_iter": meta.get("iter", "0"),
        "beam_slurm_job": bid,
    }
    csv_rows, changed = upsert(csv_rows, (model_key, hts), new_row)
    if changed:
        backfilled += 1
        print(f"  back-filled CSV: {model_key} @ {hts}  greedy={gc}/{gt}  beam7={bc}/{bt}")
    # analysis CSV
    adir = os.path.join(repo, "data_storage/things_on_paper/monitor",
                        f"analysis_d2_arch_{D}_l6_best_snapshot_{ts_c}")
    arow = extract_analysis(adir)
    if arow:
        arow.update({"timestamp": hts, "model": model_key,
                     "train_iter": meta.get("iter", "0"),
                     "analysis_slurm_job": aid})
        # simple append-if-missing (no update-in-place for analysis)
        existing = {(r.get("model",""), r.get("timestamp","")) for r in acsv_rows}
        if (model_key, hts) not in existing:
            acsv_rows.append(arow)

if backfilled:
    write_csv_rows(csv_path, csv_fields, csv_rows)
    print(f"  back-fill summary: wrote {backfilled} updated row(s) → {os.path.basename(csv_path)}")
# always rewrite acsv (it's small); harmless no-op if unchanged
if any(acsv_rows):
    write_csv_rows(acsv_path, acsv_fields, acsv_rows)
PY

# --- 0b. preserve best-ever snapshots (using only completed evals) ---
# Scans CSV for each model's best-beam7 row whose beam7_total == BEAM_MAX_TEST
# (i.e. the eval ran to completion, not a partial mid-flush number). If the
# matching snapshot .pt still exists and its metrics exceed the currently
# preserved _snapshot_best.pt, copy + rewrite the sidecar. Must run BEFORE
# rotation (step 1a) so late-completed snapshots get preserved before they're
# eligible for deletion.
BEAM_MAX_TEST_FOR_BEST=60
python3 - "$REPO/data_storage/things_on_paper/model" "$CSV" "$BEAM_MAX_TEST_FOR_BEST" <<'PY' || true
import csv, os, shutil, sys
model_dir, csv_path, cap_s = sys.argv[1:]
CAP = int(cap_s)
if not os.path.isfile(csv_path):
    sys.exit(0)

best_per_model: dict[str, dict] = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        try:
            bt = int(float(row.get("beam7_total") or 0))
            bc = int(float(row.get("beam7_correct") or 0))
        except ValueError:
            continue
        if bt != CAP:  # ignore partial / mid-run rows
            continue
        acc = bc / bt if bt else 0.0
        m = row["model"]
        cur = best_per_model.get(m)
        if cur is None or acc > cur["_acc"]:
            r = dict(row); r["_acc"] = acc; r["_bc"] = bc; r["_bt"] = bt
            best_per_model[m] = r

preserved = 0
for m, row in best_per_model.items():
    D = m.split("=")[1]
    ts_compact = row["timestamp"].replace("-", "").replace(" ", "_").replace(":", "")
    src_name = f"d2_arch_{D}_l6_best_snapshot_{ts_compact}.pt"
    src = os.path.join(model_dir, src_name)
    if not os.path.isfile(src):
        continue   # snapshot already rotated — can't preserve
    dst = os.path.join(model_dir, f"d2_arch_{D}_l6_snapshot_best.pt")
    sidecar = dst[:-3] + ".jobs"
    # skip if sidecar already names this exact (ts, iter)
    if os.path.isfile(sidecar):
        cur = {}
        for line in open(sidecar):
            line = line.strip()
            if not line or line.startswith("#"): continue
            kv = line.split(None, 1)
            if len(kv) == 2: cur[kv[0]] = kv[1]
        if cur.get("timestamp") == row["timestamp"] and cur.get("iter") == row.get("train_iter"):
            continue
        try:
            prev_acc = float(cur.get("beam7_acc", "-1"))
        except ValueError:
            prev_acc = -1.0
        if row["_acc"] * 100 < prev_acc:
            continue  # existing sidecar claims a better row; don't downgrade
    shutil.copy2(src, dst)
    with open(sidecar, "w") as fh:
        fh.write("# best-snapshot metadata (rewritten each time beam7 ties/beats the record)\n")
        fh.write(f"timestamp {row['timestamp']}\n")
        fh.write(f"iter {row.get('train_iter', '0')}\n")
        fh.write(f"valid_loss {row.get('valid_loss', 'NaN')}\n")
        fh.write(f"greedy_acc {row.get('greedy_acc', 'NaN')}\n")
        fh.write(f"beam7_acc {row.get('beam7_acc', 'NaN')}\n")
        fh.write(f"greedy_correct {row.get('greedy_correct', '0')}/{row.get('greedy_total', '0')}\n")
        fh.write(f"beam7_correct {row['_bc']}/{row['_bt']}\n")
        fh.write(f"source_snapshot d2_arch_{D}_l6_best_snapshot_{ts_compact}\n")
    print(f"  d={D}: preserved best (beam7={row['_acc']*100:.2f}%, {row['_bc']}/{row['_bt']}) → {os.path.basename(dst)}")
    preserved += 1
if preserved:
    print(f"  preservation summary: {preserved} model(s) updated")
PY

# --- 1a. rotate old snapshots whose evaluations are terminal (done or dead) ---
# Delete a snapshot if:
#   (a) its (model, timestamp) already has a CSV row, OR
#   (b) it has no sidecar .jobs file (orphan from a crashed pass), OR
#   (c) its sidecar lists only non-running SLURM jobs — meaning the evals are
#       either completed-but-didn't-write-CSV or cancelled/failed/timed out.
# Keep the snapshot if any of its sidecar jobs is still PENDING/RUNNING.
# Never touches ``_best.pt`` or ``_snapshot_best.pt`` (pattern mismatch).
python3 - "$REPO/data_storage/things_on_paper/model" "$CSV" "$BEAM_MAX_TEST" <<'PY' || true
import csv, glob, os, re, subprocess, sys
model_dir, csv_path, beam_cap_str = sys.argv[1], sys.argv[2], sys.argv[3]
BEAM_CAP = int(beam_cap_str)   # beam7_total must equal this for "eval complete"

evaluated = set()
if os.path.isfile(csv_path):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ts = row.get("timestamp", "")
            ts_c = ts.replace("-", "").replace(" ", "_").replace(":", "")
            # Only treat a row as complete when beam7_total hit the full cap.
            # Partial rows (slow early-training beam evals that didn't finish
            # within the 15-min wait) stay ineligible for rotation so the next
            # pass's back-fill can still update them.
            try:
                bt = int(float(row.get("beam7_total") or 0))
            except (TypeError, ValueError):
                bt = 0
            if bt >= BEAM_CAP:
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

pat = re.compile(r"^d2_arch_([0-9a-z]+)_l6_best_snapshot_(\d{8}_\d{6})\.pt$")
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
for D in 768 256b 512r2 768r2; do
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
    PREV_SIDECAR=$(ls -t "$REPO/data_storage/things_on_paper/model/d2_arch_${D}_l6_best_snapshot_"*.jobs 2>/dev/null | head -1 || true)
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
BEAM_WIDTH_DEFAULT=30
# d=768 / d=768r2 OOM on 16 GiB eval GPUs at width 30 (KV-cache is cloned per
# beam candidate inside model_kvcache.py). Drop them to width 20.
declare -A BEAM_WIDTH_PER_D=(
    [256]=30 [512]=30 [768]=20
    [256b]=30 [512r2]=30 [768r2]=20
)
BEAM_MAX_TEST=60   # ±1.7pp resolution at beam width 30; 20-min cycle has headroom
for D in 768 256b 512r2 768r2; do
    [[ -z "${SNAP[$D]:-}" ]] && continue
    BW="${BEAM_WIDTH_PER_D[$D]:-$BEAM_WIDTH_DEFAULT}"
    GREEDY_JOB[$D]=$(CKPT_TAG="${SNAP[$D]}" MAX_TEST=200 sbatch --parsable \
        Training/things_on_paper/eval/run_greedy_eval.sh \
        Training/things_on_paper/configs/d2_arch_${D}_l6.env test)
    BEAM_JOB[$D]=$(CKPT_TAG="${SNAP[$D]}" MAX_TEST="$BEAM_MAX_TEST" sbatch --parsable \
        Training/things_on_paper/eval/run_beam_eval.sh \
        Training/things_on_paper/configs/d2_arch_${D}_l6.env test "$BW")
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
    echo "  d=$D submitted: greedy=${GREEDY_JOB[$D]} beam${BW}=${BEAM_JOB[$D]} analysis=${ANALYSIS_JOB[$D]}"
done

# --- 3. wait for all 9 to finish (or timeout at 15 min) ---
DEADLINE=$(( $(date +%s) + 900 ))
while (( $(date +%s) < DEADLINE )); do
    RUNNING=0
    for D in 768 256b 512r2 768r2; do  # round1 surveillance reduced to 768 only
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

for D in 768 256b 512r2 768r2; do
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
    BW="${BEAM_WIDTH_PER_D[$D]:-$BEAM_WIDTH_DEFAULT}"
    echo "  d=$D results: greedy=$gc/$gt ($gacc%)  beam${BW}(@7)=$bc/$bt ($bacc%)  valid_loss=$t_vloss @ global_iter $t_iter"
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

for D in 768 256b 512r2 768r2; do
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

# Best-snapshot preservation now lives in step 0b (above), which only
# considers completed evals (beam7_total == BEAM_MAX_TEST) and runs before
# rotation so late-completing snapshots still get preserved.

# --- 5. regenerate the plots ---
python3 "$REPO/dev/plot_snapshot_history.py" || echo "plot regeneration failed (non-fatal)"
echo "=== pass complete ==="
