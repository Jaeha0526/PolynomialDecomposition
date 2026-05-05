#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm-%j.out
# BGRPO / GRPO training wrapper. One SLURM job per variant.
#
# Usage:
#   sbatch run_bgrpo.sh <run_tag> <use_beam> <reward_type>
# Arguments:
#   run_tag       — subdirectory name under data_storage/things_on_paper/BGRPO/runs/
#   use_beam      — "true" (BGRPO) | "false" (vanilla GRPO multinomial rollout)
#   reward_type   — "simple" | "rank" | "reverse_rank"
#
# Env overrides:
#   MODEL_NAME    — default d2_arch_256_l6_snapshot_best.pt
#   CONFIG_NAME   — default d2_arch_256_l6_snapshot_best.json
#   DATASET_PATH  — default things_on_paper/dataset/d2/training_dataset.txt
#   NUM_GENS      — default 32 (beam width / #rollouts)
#   NUM_Q         — default 8 (problems per outer step)
#   NUM_ITERS     — default 5 (policy updates per outer step)
#   TOTAL_SAMPLES — default 200 (non-repeating problems, paper §3.5)
#   LR            — default 1e-5
#   KL_BETA       — default 0.01
#   CLIP_EPS      — default 0.2
#   SAVE_STEPS    — default 5
#   MAX_NEW_TOK   — default 150
#
# BGRPO's invariant (per_device_train_batch_size == num_generations) requires
# a SINGLE visible GPU — enforced by --gres=gpu:...:1. CUDA_VISIBLE_DEVICES
# is auto-populated by SLURM; do not pin manually.

set -euo pipefail

RUN_TAG="${1:?usage: $0 <run_tag> <use_beam> <reward_type>}"
USE_BEAM="${2:?usage: $0 <run_tag> <use_beam> <reward_type>}"
REWARD_TYPE="${3:?usage: $0 <run_tag> <use_beam> <reward_type>}"

# Wandb project/run — set to empty string to disable.
WANDB_PROJECT="${WANDB_PROJECT:-polydec-bgrpo}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}_$(date +%m%d_%H%M)}"

MODEL_NAME="${MODEL_NAME:-d2_arch_256_l6_snapshot_best.pt}"
CONFIG_NAME="${CONFIG_NAME:-d2_arch_256_l6_snapshot_best.json}"
NUM_GENS="${NUM_GENS:-32}"
NUM_Q="${NUM_Q:-8}"
NUM_ITERS="${NUM_ITERS:-5}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-200}"
LR="${LR:-1e-5}"
KL_BETA="${KL_BETA:-0.01}"
CLIP_EPS="${CLIP_EPS:-0.2}"
SAVE_STEPS="${SAVE_STEPS:-5}"
SAVE_AT_STEPS="${SAVE_AT_STEPS:-}"   # comma-separated list; overrides SAVE_STEPS
MAX_NEW_TOK="${MAX_NEW_TOK:-150}"
START_OUTER_STEP="${START_OUTER_STEP:-0}"
DECAY_BASE="${DECAY_BASE:-}"         # empty → python default (= num_generations)
SEED="${SEED:-148}"                  # reproducibility / prompt-shuffle ordering

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO="$(cd "$HERE/../../.." && pwd)"
fi

BGRPO_ROOT="${REPO}/data_storage/things_on_paper/BGRPO"
# MODEL_DIR default = things_on_paper/model (SFT snapshot location). Override
# via `MODEL_DIR=... sbatch run_bgrpo.sh` for continuation runs that start
# from a BGRPO checkpoint dir (containing model.pt).
MODEL_DIR="${MODEL_DIR:-${REPO}/data_storage/things_on_paper/model}"
CONFIG_DIR="${BGRPO_ROOT}/configs"
DATASET_PATH="${DATASET_PATH:-${REPO}/data_storage/things_on_paper/dataset/d2/training_dataset.txt}"
OUTPUT_DIR="${BGRPO_ROOT}/runs/${RUN_TAG}"
mkdir -p "$OUTPUT_DIR"

source "$REPO/.venv/bin/activate"

echo "+ BGRPO run=${RUN_TAG}  use_beam=${USE_BEAM}  reward=${REWARD_TYPE}"
echo "+ model=${MODEL_DIR}/${MODEL_NAME}"
echo "+ output=${OUTPUT_DIR}"

cd "${REPO}/Training/BGRPO"
python3 run_bgrpo.py \
    --model_name "$MODEL_NAME" \
    --config_name "$CONFIG_NAME" \
    --model_dir "$MODEL_DIR" \
    --config_dir "$CONFIG_DIR" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --reward_type "$REWARD_TYPE" \
    --use_beam "$USE_BEAM" \
    --num_generations "$NUM_GENS" \
    --num_questions "$NUM_Q" \
    --num_iterations "$NUM_ITERS" \
    --total_training_samples "$TOTAL_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOK" \
    --rollout_temperature 1.0 \
    --lr "$LR" \
    --kl_beta "$KL_BETA" \
    --clip_epsilon "$CLIP_EPS" \
    --save_steps "$SAVE_STEPS" \
    --save_at_steps "$SAVE_AT_STEPS" \
    --start_outer_step "$START_OUTER_STEP" \
    --seed "$SEED" \
    ${DECAY_BASE:+--decay_base "$DECAY_BASE"} \
    ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"} \
    ${WANDB_RUN_NAME:+--wandb_run_name "$WANDB_RUN_NAME"}
