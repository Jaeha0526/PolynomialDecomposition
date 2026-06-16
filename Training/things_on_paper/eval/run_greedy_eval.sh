#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
# Generic greedy (batched) evaluation wrapper.
#
# Usage:
#   sbatch eval/run_greedy_eval.sh configs/<name>.env [split]
#
# The config's DATA_TAG selects the dataset subdir; ``split`` picks which
# file inside it — one of {test, valid} (default: test). The checkpoint
# tag is ``${MODEL_TAG}_best`` (written by run_experiment.sh's Trainer).
#
# Override with env vars:
#   CKPT_TAG   — alternate checkpoint stem (default: ${MODEL_TAG}_best)
#   MAX_TEST   — cap on test examples (default: 3000)

set -euo pipefail

CONFIG="${1:?usage: $0 configs/<name>.env [split]}"
SPLIT="${2:-test}"
MAX_TEST="${MAX_TEST:-3000}"

# shellcheck disable=SC1090
source "$CONFIG"
: "${BLOCK_SIZE:?}" "${N_LAYER:?}" "${N_EMBD:?}" "${DATA_TAG:?}" "${MODEL_TAG:?}"
: "${N_HEAD:=8}" "${MAX_NUMBER_TOKEN:=101}"

CKPT_TAG="${CKPT_TAG:-${MODEL_TAG}_best}"
case "$SPLIT" in
    test)  FILENAME="test_dataset.txt" ;;
    valid) FILENAME="validation_dataset.txt" ;;
    *)     FILENAME="${SPLIT}.txt" ;;
esac

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO="$(cd "$HERE/../../.." && pwd)"
fi
DATASET_DIR="${REPO}/data_storage/things_on_paper/dataset/${DATA_TAG}"
MODEL_DIR="${REPO}/data_storage/things_on_paper/model"
PRED_DIR="${REPO}/data_storage/things_on_paper/predictions"
mkdir -p "$PRED_DIR"

source "$REPO/.venv/bin/activate"

CMD=(python3 "${REPO}/Training/mingpt/run.py" inequality_evaluate4
     --block_size "$BLOCK_SIZE"
     --n_embd "$N_EMBD"
     --n_layer "$N_LAYER"
     --n_head "$N_HEAD"
     --max_number_token "$MAX_NUMBER_TOKEN"
     --max_output_length 150
     --max_test "$MAX_TEST"
     --evaluate_corpus_path "${DATASET_DIR}/${FILENAME}"
     --reading_params_path "${MODEL_DIR}/${CKPT_TAG}.pt"
     --outputs_path "${PRED_DIR}/${CKPT_TAG}__${DATA_TAG}_${SPLIT}__greedy.txt")

if [[ -n "${EXTENDED_VOCAB:-}" ]]; then
    CMD+=(--extended_vocab --sympy 1)
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
