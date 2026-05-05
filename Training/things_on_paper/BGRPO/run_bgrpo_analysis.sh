#!/bin/bash
#SBATCH --job-name=bgrpo_analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/slurm-%j.out
# Run token-category (confusion) + circuits analysis on one BGRPO checkpoint.
# Mirrors run_bgrpo_eval.sh signature so orchestrators stay symmetric.
#
# Usage:
#   sbatch run_bgrpo_analysis.sh <run_tag> <ckpt_name> [n_samples] [line_idx]
# Defaults: n_samples=200, line_idx=0.
#
# Reads BGRPO checkpoint from
#   data_storage/things_on_paper/BGRPO/runs/<run_tag>/<ckpt_name>/model.pt
# Reads architecture from
#   data_storage/things_on_paper/BGRPO/configs/d2_arch_256_l6_snapshot_best.json
# Writes JSON artefacts to
#   data_storage/things_on_paper/BGRPO/runs/<run_tag>/analysis/<ckpt_name>/{confusion,circuits}.json

set -euo pipefail

RUN_TAG="${1:?usage: $0 <run_tag> <ckpt_name> [n_samples] [line_idx]}"
CKPT_NAME="${2:?usage: $0 <run_tag> <ckpt_name> [n_samples] [line_idx]}"
N_SAMPLES="${3:-200}"
LINE_IDX="${4:-0}"

CONFIG_NAME="${CONFIG_NAME:-d2_arch_256_l6_snapshot_best.json}"
DATA_TAG="${DATA_TAG:-d2}"
DATA_FILE="${DATA_FILE:-test_dataset.txt}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
BGRPO_ROOT="${REPO}/data_storage/things_on_paper/BGRPO"
CKPT_PATH="${BGRPO_ROOT}/runs/${RUN_TAG}/${CKPT_NAME}/model.pt"
CONFIG_PATH="${BGRPO_ROOT}/configs/${CONFIG_NAME}"
DATA_PATH="${REPO}/data_storage/things_on_paper/dataset/${DATA_TAG}/${DATA_FILE}"
OUT_DIR="${BGRPO_ROOT}/runs/${RUN_TAG}/analysis/${CKPT_NAME}"

[[ -f "$CKPT_PATH"   ]] || { echo "checkpoint not found: $CKPT_PATH" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "config not found: $CONFIG_PATH" >&2; exit 1; }
mkdir -p "$OUT_DIR"

BLOCK_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['block_size'])")
N_LAYER=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['n_layer'])")
N_HEAD=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['n_head'])")
N_EMBD=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['n_embd'])")
MAX_NUMBER_TOKEN=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print(c.get('MAX_NUMBER_TOKEN', 101))")

cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO/Training:${PYTHONPATH:-}"

COMMON=( --model_path "$CKPT_PATH" --test_corpus "$DATA_PATH"
         --block_size "$BLOCK_SIZE" --n_layer "$N_LAYER"
         --n_head "$N_HEAD" --n_embd "$N_EMBD"
         --extended_vocab --max_number_token "$MAX_NUMBER_TOKEN" )

echo "=== analysis: run=${RUN_TAG} ckpt=${CKPT_NAME} ==="
echo "  ckpt:   $CKPT_PATH"
echo "  data:   $DATA_PATH"
echo "  output: $OUT_DIR"

echo "=== confusion (n=$N_SAMPLES) ==="
python3 -m analysis confusion "${COMMON[@]}" \
    --n_samples "$N_SAMPLES" --out_json "$OUT_DIR/confusion.json"

echo "=== circuits (line $LINE_IDX) ==="
python3 -m analysis circuits "${COMMON[@]}" \
    --line_idx "$LINE_IDX" --out_json "$OUT_DIR/circuits.json"

echo "=== done ==="
