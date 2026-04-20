#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
# Generic training wrapper for paper experiments.
#
# Usage:
#   sbatch run_experiment.sh configs/<name>.env
#
# The referenced .env file must `export` the following variables:
#   DATA_TAG            dataset prefix under ../../data_storage/things_on_paper/dataset/
#   MODEL_TAG           model prefix under ../../data_storage/things_on_paper/model/
#   EXP_NAME            wandb run name
#   BLOCK_SIZE          transformer context window
#   N_LAYER, N_HEAD, N_EMBD   architecture hyperparams
#   MAX_NUMBER_TOKEN    (default 101)
#   BATCH_SIZE, NUM_EPOCHS, ITERATION_PERIOD, FINETUNE_LR    training hyperparams
#   EXTENDED_VOCAB      "true" for multi-variable, empty otherwise
#   READ_CKPT           optional absolute-or-relative checkpoint to resume from
#
# All paths resolved by this script are repo-relative so the repo is
# self-contained when cloned.

set -euo pipefail

CONFIG="${1:?usage: $0 configs/<name>.env}"
if [[ ! -f "$CONFIG" ]]; then
    echo "config file not found: $CONFIG" >&2
    exit 1
fi

# Capture any user-supplied overrides BEFORE sourcing the config — otherwise
# an ``export NUM_EPOCHS=5`` (or similar) inside the config silently clobbers
# ``NUM_EPOCHS=7 sbatch …`` style overrides at the call site.
_OVERRIDE_NAMES=(NUM_EPOCHS BATCH_SIZE ITERATION_PERIOD FINETUNE_LR EXP_NAME READ_CKPT)
declare -A _OVERRIDES
for _k in "${_OVERRIDE_NAMES[@]}"; do
    if [[ -n "${!_k:-}" ]]; then
        _OVERRIDES[$_k]="${!_k}"
    fi
done

# shellcheck disable=SC1090
source "$CONFIG"

# Re-apply user overrides on top of whatever the config exported.
for _k in "${!_OVERRIDES[@]}"; do
    eval "$_k=\"\${_OVERRIDES[\$_k]}\""
done

: "${DATA_TAG:?DATA_TAG not set in $CONFIG}"
: "${MODEL_TAG:?MODEL_TAG not set in $CONFIG}"
: "${EXP_NAME:?EXP_NAME not set in $CONFIG}"
: "${BLOCK_SIZE:?BLOCK_SIZE not set}"
: "${N_LAYER:?N_LAYER not set}"
: "${N_HEAD:=8}"
: "${N_EMBD:?N_EMBD not set}"
: "${MAX_NUMBER_TOKEN:=101}"
: "${BATCH_SIZE:=256}"
: "${NUM_EPOCHS:=5}"
: "${ITERATION_PERIOD:=3000}"
: "${FINETUNE_LR:=0.0006}"
: "${EXTENDED_VOCAB:=}"

# sbatch copies the script to /var/spool/slurmd/; anchor on SLURM_SUBMIT_DIR
# (the cwd at submit time, by convention the repo root).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
    HERE="$REPO/Training/things_on_paper"
else
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO="$(cd "$HERE/../.." && pwd)"
fi
DATASET_DIR="${REPO}/data_storage/things_on_paper/dataset/${DATA_TAG}"
MODEL_DIR="${REPO}/data_storage/things_on_paper/model"
# Per-round subdir (e.g. ROUND_DIR=round3) so distinct experiment cohorts don't
# commingle under model/. Default empty = flat layout (round1/round2 today).
if [[ -n "${ROUND_DIR:-}" ]]; then
    MODEL_DIR="${MODEL_DIR}/${ROUND_DIR}"
fi
mkdir -p "$MODEL_DIR"

source "$REPO/.venv/bin/activate"

CMD=(python3 "${REPO}/Training/mingpt/run.py" inequality_finetune
     --block_size "$BLOCK_SIZE"
     --num_epochs "$NUM_EPOCHS"
     --n_embd "$N_EMBD"
     --n_layer "$N_LAYER"
     --n_head "$N_HEAD"
     --max_number_token "$MAX_NUMBER_TOKEN"
     --iteration_period "$ITERATION_PERIOD"
     --lr_decay 1
     --shuffle 1
     --batch_size "$BATCH_SIZE"
     --finetune_lr "$FINETUNE_LR"
     --valid_corpus_path "${DATASET_DIR}/validation_dataset.txt"
     --evaluate_corpus_path "${DATASET_DIR}/test_dataset.txt"
     --finetune_corpus_path "${DATASET_DIR}/training_dataset.txt"
     --writing_params_path "${MODEL_DIR}/${MODEL_TAG}.pt"
     --dataset_name "$DATA_TAG"
     --exp_name "$EXP_NAME")

if [[ -n "$EXTENDED_VOCAB" ]]; then
    CMD+=(--extended_vocab)
fi

if [[ -n "${READ_CKPT:-}" ]]; then
    CMD+=(--reading_params_path "$READ_CKPT")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
