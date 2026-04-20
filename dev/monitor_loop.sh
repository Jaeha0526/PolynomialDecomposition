#!/bin/bash
# Outer loop: call dev/snapshot_eval_one_pass.sh every ~15 min until the
# three SFT training jobs are all done. Self-terminates cleanly when there
# are no more ``run_experiment.sh`` jobs in the queue for this user.

set -uo pipefail

REPO="/resnick/groups/Hippo/jaeha/PolynomialDecomposition"
cd "$REPO"
LOG="$REPO/data_storage/things_on_paper/monitor/monitor.log"
mkdir -p "$(dirname "$LOG")"
PIDFILE="$REPO/data_storage/things_on_paper/monitor/monitor.pid"
INTERVAL=${INTERVAL:-1200}  # seconds (20 min default)

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

    # Fast exit if training is done (no more run_experiment.sh jobs).
    RUNNING_TRAIN=$(squeue -u "$USER" -h -n run_experiment.sh | wc -l)
    if [[ "$RUNNING_TRAIN" -eq 0 ]]; then
        echo "$(date) — no run_experiment.sh jobs in queue; exiting loop" >> "$LOG"
        break
    fi

    bash "$REPO/dev/snapshot_eval_one_pass.sh" >> "$LOG" 2>&1 || {
        echo "$(date) — one_pass failed, continuing" >> "$LOG"
    }

    # Sleep until the next 15-min boundary, minus the time we already spent.
    ELAPSED=$(( $(date +%s) - START ))
    REMAIN=$(( INTERVAL - ELAPSED ))
    if (( REMAIN > 0 )); then
        sleep "$REMAIN"
    fi
done

# Final plot regen once training is over (training logs may have grown).
python3 "$REPO/dev/plot_snapshot_history.py" >> "$LOG" 2>&1 || true
echo "monitor loop done at $(date)" >> "$LOG"
