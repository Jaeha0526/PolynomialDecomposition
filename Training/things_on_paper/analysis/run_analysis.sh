#!/bin/bash
#SBATCH --job-name=paper_analysis
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#
# Run one of the paper-analysis modes on a D2 checkpoint.
#
#   sbatch run_analysis.sh <config.env> [mode] [extra args...]
#
# Arguments:
#   <config.env>   — Training/things_on_paper/configs/d2_arch_*.env (for model shape)
#   [mode]         — one of confusion|circuits|viz  (default: confusion)
#
# Environment overrides:
#   CKPT_TAG       — alternate checkpoint stem (default: ${MODEL_TAG}_best)
#   DATA_FILE      — test/validation file under dataset/${DATA_TAG}/ (default: test_dataset.txt)
#   N_SAMPLES      — for confusion mode (default: 1000, matches paper)
#   LINE_IDX       — for circuits/viz mode (default: 0)
#   OUT_DIR        — for viz mode (default: dev/plots/analysis_${MODEL_TAG}_${CKPT_TAG})
set -euo pipefail

CONFIG="${1:?usage: sbatch run_analysis.sh <config.env> [mode] [extra args...]}"
MODE="${2:-confusion}"
shift 2 2>/dev/null || true

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO/Training:${PYTHONPATH:-}"

source "$CONFIG"
: "${CKPT_TAG:=${MODEL_TAG}_best}"
: "${DATA_FILE:=test_dataset.txt}"
: "${N_SAMPLES:=1000}"
: "${LINE_IDX:=0}"

MODEL="$REPO/data_storage/things_on_paper/model/${CKPT_TAG}.pt"
DATA="$REPO/data_storage/things_on_paper/dataset/${DATA_TAG}/${DATA_FILE}"
OUT_DIR_DEFAULT="$REPO/dev/plots/analysis_${MODEL_TAG}_${CKPT_TAG}"
: "${OUT_DIR:=$OUT_DIR_DEFAULT}"

COMMON=( --model_path "$MODEL" --test_corpus "$DATA"
         --block_size "$BLOCK_SIZE" --n_layer "$N_LAYER"
         --n_head "$N_HEAD" --n_embd "$N_EMBD" )
if [[ "${EXTENDED_VOCAB:-false}" == "true" ]]; then
    COMMON+=( --extended_vocab --max_number_token "${MAX_NUMBER_TOKEN:-101}" )
fi

case "$MODE" in
    confusion)
        OUT_JSON="$OUT_DIR/confusion.json"
        mkdir -p "$OUT_DIR"
        python3 -m analysis confusion "${COMMON[@]}" \
            --n_samples "$N_SAMPLES" --out_json "$OUT_JSON" "$@"
        ;;
    circuits)
        OUT_JSON="$OUT_DIR/circuits_line${LINE_IDX}.json"
        mkdir -p "$OUT_DIR"
        python3 -m analysis circuits "${COMMON[@]}" \
            --line_idx "$LINE_IDX" --out_json "$OUT_JSON" "$@"
        ;;
    viz)
        mkdir -p "$OUT_DIR"
        python3 -m analysis viz "${COMMON[@]}" \
            --line_idx "$LINE_IDX" --out_dir "$OUT_DIR" "$@"
        ;;
    *)
        echo "unknown mode: $MODE  (choose confusion|circuits|viz)" >&2
        exit 2
        ;;
esac
echo "done — results under $OUT_DIR"
