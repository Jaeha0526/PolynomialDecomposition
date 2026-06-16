#!/bin/bash
# Submit a beam-width sweep matching Fig 8's x-axis.
#
# Usage:
#   bash eval/sweep_beam_widths.sh configs/<name>.env [split]
#
# Submits one SLURM job per beam width in the default sweep {1, 5, 10, 15,
# 20, 25, 30}. Override the widths by setting BEAM_WIDTHS, e.g.
#     BEAM_WIDTHS="1 7 30" bash eval/sweep_beam_widths.sh configs/... test
#
# Jobs run in parallel on separate H200 nodes (each one consumes one GPU).
# Each writes its predictions + per-width stats to data_storage/.../predictions/.

set -euo pipefail

CONFIG="${1:?usage: $0 configs/<name>.env [split]}"
SPLIT="${2:-test}"
BEAM_WIDTHS="${BEAM_WIDTHS:-1 5 10 15 20 25 30}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for w in $BEAM_WIDTHS; do
    echo "Submitting beam width $w ..."
    sbatch "${HERE}/run_beam_eval.sh" "$CONFIG" "$SPLIT" "$w"
done

echo
echo "All beam-width jobs queued. Check status with: squeue -u \$USER"
