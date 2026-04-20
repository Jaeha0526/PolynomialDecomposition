# `Training/analysis/` — paper-appendix analysis tools

Three things this package gives you, all built on a trained checkpoint:

1. **Confusion per token category** (paper Appendix C, Table 4)
   — mean predicted-target probability and greedy accuracy split by
   SIGN / OPERATOR / NUMBER / VARIABLE / DELIMITER. The paper's finding is
   that SIGN tokens are near-random (~52%) while operators and numbers sit
   around 94% / 90%; that's the quantitative justification for beam
   search.

2. **Attention-circuit detectors** (paper Appendix D, "Monomial Heads")
   — per-(layer, head) scores for three specific circuits:

   | Score | Paper pattern |
   |---|---|
   | `previous_token` | Layer-0 heads that attend 1-5 positions behind. |
   | `within_monomial` | Layer-1 heads that attend within the same `+`-delimited segment. |
   | `delimiter` | Layer-1 heads on the answer side that focus on `&` tokens. |

3. **Figure reproducers**
   — `fig9_top3.png` (top-3 probabilities per answer position) and
   `fig10_*.png` (attention heatmap of the top-scoring head for the
   chosen circuit).

## Quickstart

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/Training:${PYTHONPATH:-}"

# Table 4 reproduction — 1000 test lines, any GPU, ~30s on H200
python3 -m analysis confusion \
  --model_path data_storage/things_on_paper/model/d2_arch_512_l6_best.pt \
  --test_corpus data_storage/things_on_paper/dataset/d2/test_dataset.txt \
  --block_size 850 --n_layer 6 --n_head 8 --n_embd 512 \
  --extended_vocab --max_number_token 101 \
  --n_samples 1000 --out_json dev/plots/confusion_d512.json

# Appendix D circuit detection (single line)
python3 -m analysis circuits  [same flags] \
  --line_idx 0 --out_json dev/plots/circuits_d512_l0.json

# All three figures in one go
python3 -m analysis viz  [same flags] \
  --line_idx 0 --out_dir dev/plots/viz_d512_l0 --which_attn previous_token
```

Under SLURM, use the wrapper — it sources the model config for you:

```bash
# confusion
sbatch Training/things_on_paper/analysis/run_analysis.sh \
  Training/things_on_paper/configs/d2_arch_512_l6.env confusion

# circuits at line 7
LINE_IDX=7 sbatch Training/things_on_paper/analysis/run_analysis.sh \
  Training/things_on_paper/configs/d2_arch_512_l6.env circuits

# figures
OUT_DIR=dev/plots/figs_d512 sbatch Training/things_on_paper/analysis/run_analysis.sh \
  Training/things_on_paper/configs/d2_arch_512_l6.env viz --which_attn delimiter
```

## Interpreting the circuit scores

Scores are in [0, 1] and are averages of attention mass over the valid rows.
Higher = more-paper-like. On a freshly trained 6-layer d=512 model we see
typical patterns like:

- `previous_token_top[0]`  ≈ **(L0, H7, 0.87)** — the paper's canonical
  previous-token head (Fig 10 left).
- `delimiter_top[0]`       ≈ **(L1, H6, 0.68)** — the `&`-delimiter head
  (Fig 10 right).

If your model does *not* exhibit these patterns (scores < 0.3) while still
achieving paper-level accuracy, that is itself an interesting result.

## Tests

```bash
bash Training/analysis/run_tests.sh   # 9 pure-CPU unit tests
```

## File map

| File | Purpose |
|---|---|
| `token_category.py` | String → TokenCategory classifier; `category_map(itos)` tensor |
| `confusion.py`      | Table-4 reproduction (CLI + library) |
| `circuits.py`       | Attention capture + 3 circuit scorers (CLI + library) |
| `viz.py`            | Fig 9 / Fig 10 / circuit-score heatmap (CLI + library) |
| `__main__.py`       | `python -m analysis {confusion|circuits|viz}` dispatcher |
| `tests/`            | Unit tests (no checkpoint required) |
| `run_tests.sh`      | Shell wrapper for the unit tests |
