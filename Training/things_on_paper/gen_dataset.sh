#!/bin/bash
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
# Parallel dataset generation on a CPU compute node.
#
# Usage:
#   sbatch Training/things_on_paper/gen_dataset.sh configs/<name>.env
#
# The config file's variables (DATA_TAG, paper-axis spec) drive the call.
# See configs/README.md for the paper-section → config map.
#
# This runs using_sympy on ${SLURM_CPUS_PER_TASK} workers. At ~70 samples/sec
# per worker, 2M examples takes ~8 h on 64 cores.

set -euo pipefail

CONFIG="${1:?usage: $0 configs/<name>.env}"
if [[ ! -f "$CONFIG" ]]; then
    echo "config file not found: $CONFIG" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"

: "${DATA_TAG:?DATA_TAG not set in $CONFIG}"

# Fixed across paper experiments; paper §3.1 / Table 1.
: "${NUM_INNER_VARS:=3}"
: "${NUM_OUTER_VARS:=3}"
: "${MAX_DEGREE_INNER:=3}"
: "${MAX_DEGREE_OUTER:=3}"
: "${MAX_TERMS_INNER:=3}"
: "${MAX_TERMS_OUTER:=3}"
# Paper D2 = (-5, 5); D3-adapt uses (-20, 20) inner and varies outer.
: "${COEFF_MIN_INNER:=-5}"
: "${COEFF_MAX_INNER:=5}"
: "${COEFF_MIN_OUTER:=-5}"
: "${COEFF_MAX_OUTER:=5}"
: "${NUM_TRAIN:=2000000}"
: "${NUM_TEST:=3000}"
: "${NUM_VALID:=1000}"

# When run under sbatch, ${BASH_SOURCE[0]} points to /var/spool/slurmd/... —
# SLURM copies the script there. Anchor on SLURM_SUBMIT_DIR (the cwd when
# sbatch was invoked) and assume sbatch is run from the repo root. Falls
# back to ${BASH_SOURCE[0]} when the script is invoked directly.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO="$(cd "$HERE/../.." && pwd)"
fi
OUT="$REPO/data_storage/things_on_paper/dataset/${DATA_TAG}"

# Inherit SLURM's CPU allocation; fall back to auto-detect if run outside SLURM.
export WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export REPO

source "$REPO/.venv/bin/activate"

# Dispatch to a real .py file (not a heredoc) so multiprocessing spawn can
# re-import the main module in workers. The driver reads its config from
# the environment variables we exported above.
export NUM_INNER_VARS NUM_OUTER_VARS MAX_DEGREE_INNER MAX_DEGREE_OUTER
export MAX_TERMS_INNER MAX_TERMS_OUTER
export COEFF_MIN_INNER COEFF_MAX_INNER COEFF_MIN_OUTER COEFF_MAX_OUTER
export NUM_TRAIN NUM_TEST NUM_VALID DATA_TAG

python3 "$REPO/Training/things_on_paper/_gen_dataset.py"
