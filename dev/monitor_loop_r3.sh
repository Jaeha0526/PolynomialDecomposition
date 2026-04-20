#!/bin/bash
# Outer loop for the round-3 cohort (d=256r3 / d=512r3). Calls
# snapshot_eval_one_pass_r3.sh every ~20 min, terminates when the two
# specific round-3 jobs leave the queue. Separate PID / log so it runs
# alongside the existing r1/r2 monitor loop without interference.

set -uo pipefail

REPO="/resnick/groups/Hippo/jaeha/PolynomialDecomposition"
cd "$REPO"
ROUND_DIR_R3="$REPO/data_storage/things_on_paper/monitor/round3"
LOG="$ROUND_DIR_R3/monitor.log"
mkdir -p "$(dirname "$LOG")"
PIDFILE="$ROUND_DIR_R3/monitor.pid"
INTERVAL=${INTERVAL:-1200}  # seconds (20 min default)
# Round-3 training jobs are tracked in train_chain_r3.txt — we exit only once
# both have left the queue.
R3_JOB_NAMES_GREP="run_experiment.sh"

# --- singleton guard + session detach ---
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "monitor already running (pid $(cat "$PIDFILE")); exiting."
    exit 0
fi
# Detach from the controlling terminal so Claude-Code/SSH exits don't SIGHUP us.
if [[ -z "${MONITOR_DETACHED:-}" ]]; then
    MONITOR_DETACHED=1 nohup setsid bash "$0" "$@" >/dev/null 2>&1 &
    echo "monitor loop detached (pid $!)"
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

echo "monitor loop started (pid $$) at $(date)" >> "$LOG"

while true; do
    START=$(date +%s)

    # Fast exit when neither round-3 training job is in the queue.
    R3_JOBS=$(squeue -u "$USER" -h -n run_experiment.sh -o "%i %j" \
        | awk '/run_experiment/ {print $1}' | wc -l)
    CHAIN_FILE="$ROUND_DIR_R3/train_chain.txt"
    if [[ -f "$CHAIN_FILE" ]]; then
        # Exit only when every job id listed in round3's chain file is terminal.
        IDS=$(awk '!/^#/ {for (i=2;i<=NF;i++) print $i}' "$CHAIN_FILE" | sort -u | tr '\n' ',')
        if [[ -n "$IDS" ]]; then
            ACTIVE=$(squeue -u "$USER" -h -j "${IDS%,}" -o "%i" 2>/dev/null | wc -l)
            if (( ACTIVE == 0 )); then
                echo "$(date) — no round-3 training jobs active; exiting loop" >> "$LOG"
                break
            fi
        fi
    elif [[ "$R3_JOBS" -eq 0 ]]; then
        echo "$(date) — no run_experiment.sh jobs in queue and no chain file; exiting" >> "$LOG"
        break
    fi

    bash "$REPO/dev/snapshot_eval_one_pass_r3.sh" >> "$LOG" 2>&1 || {
        echo "$(date) — one_pass_r3 failed, continuing" >> "$LOG"
    }

    # Sleep until the next 15-min boundary, minus the time we already spent.
    ELAPSED=$(( $(date +%s) - START ))
    REMAIN=$(( INTERVAL - ELAPSED ))
    if (( REMAIN > 0 )); then
        sleep "$REMAIN"
    fi
done

# Final plot regen once training is over (training logs may have grown).
python3 "$REPO/dev/plot_snapshot_history_r3.py" >> "$LOG" 2>&1 || true
echo "monitor loop done at $(date)" >> "$LOG"
