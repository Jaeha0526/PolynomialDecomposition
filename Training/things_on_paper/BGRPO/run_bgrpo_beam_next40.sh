#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm-%j.out
# Run beam eval ONLY on problems 60-99 (the "next 40" after the first 60).
# Used to extend existing beam-60 eval summaries to beam-100 coverage without
# re-running the already-completed first 60 problems.
#
# Usage:
#   sbatch run_bgrpo_beam_next40.sh <run_tag> <ckpt_name> [beam_width]
#
# Writes:
#   runs/<run_tag>/eval/<ckpt_name>/beam_next40_predictions.txt
#   runs/<run_tag>/eval/<ckpt_name>/beam_next40.out
#   runs/<run_tag>/eval/<ckpt_name>/summary_next40.json   (beam-only)
#
# Dataset:
#   data_storage/things_on_paper/dataset/d2_test_60to99/test_dataset.txt (40 lines)

set -euo pipefail

RUN_TAG="${1:?usage: $0 <run_tag> <ckpt_name> [beam_width]}"
CKPT_NAME="${2:?usage: $0 <run_tag> <ckpt_name> [beam_width]}"
BEAM_WIDTH="${3:-30}"

CONFIG_NAME="${CONFIG_NAME:-d2_arch_256_l6_snapshot_best.json}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO="$(cd "$HERE/../../.." && pwd)"
fi

BGRPO_ROOT="${REPO}/data_storage/things_on_paper/BGRPO"
RUN_DIR="${BGRPO_ROOT}/runs/${RUN_TAG}"
CKPT_PATH="${RUN_DIR}/${CKPT_NAME}/model.pt"
OUT_DIR="${RUN_DIR}/eval/${CKPT_NAME}"
CONFIG_PATH="${BGRPO_ROOT}/configs/${CONFIG_NAME}"
DATASET_FILE="${REPO}/data_storage/things_on_paper/dataset/d2_test_60to99/test_dataset.txt"

[[ -f "$CKPT_PATH"    ]] || { echo "checkpoint not found: $CKPT_PATH" >&2; exit 1; }
[[ -f "$CONFIG_PATH"  ]] || { echo "config not found: $CONFIG_PATH" >&2; exit 1; }
[[ -f "$DATASET_FILE" ]] || { echo "dataset not found: $DATASET_FILE" >&2; exit 1; }
mkdir -p "$OUT_DIR"

BLOCK_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['block_size'])")
N_LAYER=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['n_layer'])")
N_HEAD=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['n_head'])")
N_EMBD=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['n_embd'])")
MAX_NUMBER_TOKEN=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print(c.get('MAX_NUMBER_TOKEN', 101))")

source "$REPO/.venv/bin/activate"

echo "=== BGRPO beam-next40 eval: run=${RUN_TAG} ckpt=${CKPT_NAME} ==="
echo "  ckpt:    $CKPT_PATH"
echo "  dataset: $DATASET_FILE  (40 problems, indices 60-99)"
echo "  beam_width=$BEAM_WIDTH"
echo

python3 "${REPO}/Training/mingpt/run.py" debug_beam \
    --block_size "$BLOCK_SIZE" \
    --n_embd "$N_EMBD" \
    --n_layer "$N_LAYER" \
    --n_head "$N_HEAD" \
    --max_number_token "$MAX_NUMBER_TOKEN" \
    --max_output_length 150 \
    --beam_width "$BEAM_WIDTH" \
    --max_test 40 \
    --check_path "${REPO}/Training/mingpt/check.m" \
    --evaluate_corpus_path "$DATASET_FILE" \
    --reading_params_path "$CKPT_PATH" \
    --outputs_path "${OUT_DIR}/beam_next40_predictions.txt" \
    --extended_vocab --sympy 1 \
    2>&1 | tee "${OUT_DIR}/beam_next40.out"

python3 - "$OUT_DIR" "$BEAM_WIDTH" <<'PY'
import json, re, sys
from pathlib import Path
out_dir = Path(sys.argv[1]); bw = int(sys.argv[2])
summary = {"run_tag": out_dir.parent.name, "ckpt": out_dir.name, "beam_width": bw,
           "split": "next40_problems_60_to_99"}
b_text = (out_dir / "beam_next40.out").read_text(errors="ignore")
beam_by_width = {}
for line in b_text.splitlines():
    m = re.match(r"^Beam width (\d+):\s+(\d+) out of (\d+):", line)
    if m:
        w, c, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
        beam_by_width[w] = (c, t)
summary["beam"] = {
    str(w): {"correct": c, "total": t,
             "acc": round(100 * c / t, 3) if t else 0.0}
    for w, (c, t) in sorted(beam_by_width.items())
}
(out_dir / "summary_next40.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote {out_dir / 'summary_next40.json'}")
PY

echo "=== done ==="
