#!/bin/bash
#SBATCH --job-name=paper_analysis_combo
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#
# Run both confusion (Table 4) and circuits (Appendix D) for one checkpoint
# in a single SLURM job and write JSON artefacts to $OUT_DIR.
#
# Arguments:   sbatch run_analysis_combined.sh <config.env>
# Env vars:
#   CKPT_TAG    — alternate checkpoint stem   (default: ${MODEL_TAG}_best)
#   DATA_FILE   — test/validation file       (default: test_dataset.txt)
#   N_SAMPLES   — confusion sample count     (default: 200)
#   LINE_IDX    — circuits line index        (default: 0)
#   OUT_DIR     — output directory           (default: dev/plots/analysis_${MODEL_TAG}_${CKPT_TAG})
set -euo pipefail

CONFIG="${1:?usage: sbatch run_analysis_combined.sh <config.env>}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO/Training:${PYTHONPATH:-}"

source "$CONFIG"
: "${CKPT_TAG:=${MODEL_TAG}_best}"
: "${DATA_FILE:=test_dataset.txt}"
: "${N_SAMPLES:=200}"
: "${LINE_IDX:=0}"

MODEL="$REPO/data_storage/things_on_paper/model/${CKPT_TAG}.pt"
DATA="$REPO/data_storage/things_on_paper/dataset/${DATA_TAG}/${DATA_FILE}"
: "${OUT_DIR:=$REPO/dev/plots/analysis_${MODEL_TAG}_${CKPT_TAG}}"
mkdir -p "$OUT_DIR"

COMMON=( --model_path "$MODEL" --test_corpus "$DATA"
         --block_size "$BLOCK_SIZE" --n_layer "$N_LAYER"
         --n_head "$N_HEAD" --n_embd "$N_EMBD" )
if [[ "${EXTENDED_VOCAB:-false}" == "true" ]]; then
    COMMON+=( --extended_vocab --max_number_token "${MAX_NUMBER_TOKEN:-101}" )
fi

echo "=== confusion (n=$N_SAMPLES) ==="
python3 -m analysis confusion "${COMMON[@]}" \
    --n_samples "$N_SAMPLES" --out_json "$OUT_DIR/confusion.json"

echo "=== circuits (line $LINE_IDX) ==="
python3 -m analysis circuits "${COMMON[@]}" \
    --line_idx "$LINE_IDX" --out_json "$OUT_DIR/circuits.json"

echo "done — results in $OUT_DIR"
