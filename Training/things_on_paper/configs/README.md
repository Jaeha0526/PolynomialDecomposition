# things_on_paper / configs

One `.env` file per paper training sweep. The generic runner
`run_experiment.sh` sources the chosen file and dispatches to
`../mingpt/run.py inequality_finetune` with the right flags.

## Paper → config map

| Paper section | Config | What it trains |
|---|---|---|
| §4.1 D1-a (degree scaling) | `d1_degree.env` | Single model on 2M examples, mixed `(d_inner, d_outer)` |
| §4.1 D1-b (variable scaling) | `d1_var_<v_in>_<v_out>.env` | One model per `(v_inner, v_outer)` combo, `{2,3,4}²` with the other axis fixed at 3 |
| §4.2 D2-a (arch) | `d2_arch_<d>_l<L>.env` | One model per `(n_embd, n_layer)` — covers 256/512/768 × 4/6 layers |
| §4.2 D2-b (heads) | `d2_heads_<h>.env` | One model per `n_head ∈ {4, 8, 16}` (`n_head=8` is the reuse of `d2_arch_512_l6`) |
| §4.3 D3 (adaptation) | — | Pre-train via `d2_arch_512_l6.env`, then fine-tune with `READ_CKPT=...` pointing to `_best.pt`; uses a different `DATA_TAG` for the `C2` range |

The remaining D1-b variants (`d1_var_2_3`, `d1_var_4_3`, `d1_var_3_2`,
`d1_var_3_4`) follow the same pattern as `d1_var_3_3.env` with the
corresponding `DATA_TAG` / `MODEL_TAG` / `EXP_NAME` / and appropriate
`BLOCK_SIZE`. Add as you need them — the templates are trivially
copy-editable.

## Usage

Training:

```bash
sbatch run_experiment.sh configs/d2_arch_512_l6.env
```

Greedy eval on a test split:

```bash
sbatch eval/run_greedy_eval.sh configs/d2_arch_512_l6.env model3_best data3_test
```

Beam eval (width 30, used for Fig 8 accuracy numbers):

```bash
sbatch eval/run_beam_eval.sh configs/d2_arch_512_l6.env model3_best data3_test 30
```

## Relationship to the legacy `exp*.sh` files

The legacy `things_on_paper/exp*.sh` scripts are kept for historical
reference — they are the command logs of the actual paper runs. They
have been patched (`nanogpt` → `mingpt`) but most lines are commented
out as version-controlled history. For new runs, prefer the
`configs/` + `run_experiment.sh` system.

`things_on_paper/exp_model_{0,1}.sh` reference a `run_model_0.py`
script that is not in this repo; they are left untouched and should be
ignored. D3 (distribution adaptation) can be reproduced today by
running `run_experiment.sh` twice: once on the `C1` corpus, and again
on the `C2` corpus with `READ_CKPT` pointing to the first run's
`_best.pt`.
