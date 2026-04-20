#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
# Evaluate one BGRPO checkpoint (greedy + beam sweep) and write results under
#   data_storage/things_on_paper/BGRPO/runs/<run_tag>/eval/<ckpt_name>/
#
# Usage:
#   sbatch run_bgrpo_eval.sh <run_tag> <ckpt_name> [beam_width] [greedy_max_test] [beam_max_test]
# Example:
#   sbatch run_bgrpo_eval.sh grpo checkpoint-final
#   sbatch run_bgrpo_eval.sh bgrpo_rank checkpoint-20 20 200 60
#
# Defaults:
#   beam_width=30 (raise for more precision; use 20 for d=768-class models)
#   greedy_max_test=200
#   beam_max_test=60
#
# Config comes from data_storage/things_on_paper/BGRPO/configs/ (or --config_dir
# override), default d2_arch_256_l6_snapshot_best.json (matches the SFT snapshot
# the BGRPO run was initialized from).

set -euo pipefail

RUN_TAG="${1:?usage: $0 <run_tag> <ckpt_name> [beam_width] [greedy_max_test] [beam_max_test]}"
CKPT_NAME="${2:?usage: $0 <run_tag> <ckpt_name> [beam_width] [greedy_max_test] [beam_max_test]}"
BEAM_WIDTH="${3:-30}"
GREEDY_MAX_TEST="${4:-200}"
BEAM_MAX_TEST="${5:-60}"

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
DATASET_DIR="${REPO}/data_storage/things_on_paper/dataset/d2"

if [[ ! -f "$CKPT_PATH" ]]; then
    echo "checkpoint not found: $CKPT_PATH" >&2; exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "config not found: $CONFIG_PATH" >&2; exit 1
fi
mkdir -p "$OUT_DIR"

# Extract architecture fields from the config JSON.
BLOCK_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['block_size'])")
N_LAYER=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['n_layer'])")
N_HEAD=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['n_head'])")
N_EMBD=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['n_embd'])")
MAX_NUMBER_TOKEN=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print(c.get('MAX_NUMBER_TOKEN', 101))")

source "$REPO/.venv/bin/activate"

echo "=== BGRPO eval: run=${RUN_TAG} ckpt=${CKPT_NAME} ==="
echo "  ckpt:    $CKPT_PATH"
echo "  config:  $CONFIG_PATH  (block_size=$BLOCK_SIZE n_layer=$N_LAYER n_head=$N_HEAD n_embd=$N_EMBD)"
echo "  dataset: $DATASET_DIR"
echo "  output:  $OUT_DIR"
echo "  beam_width=$BEAM_WIDTH  greedy_max_test=$GREEDY_MAX_TEST  beam_max_test=$BEAM_MAX_TEST"
echo

# --- greedy eval -------------------------------------------------------------
echo "=== greedy ==="
python3 "${REPO}/Training/mingpt/run.py" inequality_evaluate4 \
    --block_size "$BLOCK_SIZE" \
    --n_embd "$N_EMBD" \
    --n_layer "$N_LAYER" \
    --n_head "$N_HEAD" \
    --max_number_token "$MAX_NUMBER_TOKEN" \
    --max_output_length 150 \
    --max_test "$GREEDY_MAX_TEST" \
    --evaluate_corpus_path "${DATASET_DIR}/test_dataset.txt" \
    --reading_params_path "$CKPT_PATH" \
    --outputs_path "${OUT_DIR}/greedy_predictions.txt" \
    --extended_vocab --sympy 1 \
    2>&1 | tee "${OUT_DIR}/greedy.out"

# --- beam eval ---------------------------------------------------------------
echo
echo "=== beam (width $BEAM_WIDTH) ==="
python3 "${REPO}/Training/mingpt/run.py" debug_beam \
    --block_size "$BLOCK_SIZE" \
    --n_embd "$N_EMBD" \
    --n_layer "$N_LAYER" \
    --n_head "$N_HEAD" \
    --max_number_token "$MAX_NUMBER_TOKEN" \
    --max_output_length 150 \
    --beam_width "$BEAM_WIDTH" \
    --max_test "$BEAM_MAX_TEST" \
    --check_path "${REPO}/Training/mingpt/check.m" \
    --evaluate_corpus_path "${DATASET_DIR}/test_dataset.txt" \
    --reading_params_path "$CKPT_PATH" \
    --outputs_path "${OUT_DIR}/beam_predictions.txt" \
    --extended_vocab --sympy 1 \
    2>&1 | tee "${OUT_DIR}/beam.out"

# --- summary.json ------------------------------------------------------------
python3 - "$OUT_DIR" "$BEAM_WIDTH" <<'PY'
import json, re, sys
from pathlib import Path
out_dir = Path(sys.argv[1]); bw = int(sys.argv[2])
summary: dict = {"run_tag": out_dir.parent.name, "ckpt": out_dir.name, "beam_width": bw}

def last_match(pattern: str, text: str):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else None

g_text = (out_dir / "greedy.out").read_text(errors="ignore")
m = last_match(r"^Correct:\s*([\d.]+)\s+out of\s+([\d.]+)", g_text)
if m:
    correct, total = int(float(m[0])), int(float(m[1]))
    summary["greedy"] = {"correct": correct, "total": total,
                         "acc": round(100 * correct / total, 3) if total else 0.0}

b_text = (out_dir / "beam.out").read_text(errors="ignore")
beam_by_width: dict[int, tuple[int, int]] = {}
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
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote {out_dir / 'summary.json'}")
PY

echo "=== done ==="
