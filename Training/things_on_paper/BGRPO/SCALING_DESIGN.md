# (B)GRPO scaling experiments — design doc

Written 2026-04-20. Purpose: after the first round of (B)GRPO runs on the
d=256 SFT snapshot, decide how to scale further so we can compare grpo /
bgrpo-binary / bgrpo-rank on a denser learning curve and larger models.

See `bgrpo_iter_curve.png` (sibling file in `BGRPO/`) for the current 0/10/20/25-step data.

## Baseline (already done)

- Init: `data_storage/things_on_paper/model/d2_arch_256_l6_snapshot_best.pt`
  (SFT d=256, iter ≈ 56k, beam@30 = 26.67% / 60-problem test).
- Methods: vanilla GRPO, BGRPO (binary reward), BGRPO (rank-aware).
- Config: 200 non-repeating prompts, beam_width=8 during rollout,
  num_generations=8, 25 policy-update steps total, β=0.01, lr=1e-5,
  save every 5 steps.
- Wallclock: ~3 min per run on 1×H200.
- Headline @ ckpt-25: grpo 25.0% · bgrpo 28.3% · bgrpo_rank 31.7% (beam@30).

## Scaling axes

| # | Axis | Values | Per-run cost | Primary question |
|---|---|---|---|---|
| A | policy-update steps | 25 → 50 → 100 → 200 | ~linear; 200 ≈ 24 min | Does bgrpo_rank plateau, or keep widening the gap? Does vanilla GRPO eventually help? |
| B | train-time beam width | 8 → 16 → 32 | ~1.7× per doubling; KV-cache mem grows | Paper §3.5 uses 32 — does it actually matter? |
| C | init model size | d=256, d=512, d=768 | d=512 ~2×, d=768 ~4× | Does the rank-aware gap hold across model sizes? |
| D | problems / epoch | 200 → 500 → 1000 | ~linear in prompt count | Is sample efficiency the current bottleneck? |

## Recommended experiment plan

### Phase 1 — Design 1: "steps sweep" (cheap; run now)

All other config identical to baseline. Just vary training length.

```
method ∈ {grpo, bgrpo, bgrpo_rank}
steps  ∈ {50, 100, 200}
init   = d=256 snapshot_best
```

- 9 runs total · ~5 min (50) / 12 min (100) / 24 min (200) on 1×H200 each.
- Can parallelize 3 at a time (one GPU per method).
- Eval every saved checkpoint (cadence stays at every 5 steps), sweep beam
  widths 1–30 on the canonical 60-problem test set.
- Outcome: three learning curves of shape {0, 5, 10, …, 200}. Tells us where
  each method plateaus and whether the rank-aware gap widens with compute.

### Phase 2 — Design 2: "paper-faithful config" (3 runs)

Reproduce the published settings precisely so we have a direct-to-paper
data point.

```
beam_width      = 32              # paper §3.5
num_generations = 32              # = beam_width (single-GPU invariant)
num_prompts     = 200
num_updates     = 125             # = 5 × 25 batches (paper)
beta            = 0.01
lr              = 1e-5
init            = d=256 snapshot_best
```

- 3 runs (one per method) · larger beam → ~30 min each on H200.
- Comparable to paper Fig 7/8 cells (d=256, 6-layer).

### Phase 3 — Design 3: "model-size matrix" (decide after Phase 1)

Expand only the methods that looked promising in Phase 1 across d=512 and
d=768 SFT snapshots. Probable pruning:

|         | d=256 | d=512 | d=768 |
|---------|-------|-------|-------|
| grpo        | (done) | skip | skip |
| bgrpo       | (done) | 50, 200 | — |
| bgrpo_rank  | (done) | 50, 200 | 50, 200 |

- d=512 init: `model/d2_arch_512_l6_snapshot_best.pt` (iter 26k, beam7 21.4%).
- d=768 init: `model/d2_arch_768_l6_snapshot_best.pt` (iter 52k, beam7 16.7%) —
  note the d=768 snapshot_best was hand-seeded from a mid-training snapshot,
  so this run also serves as a smoke test on a weaker init.
- KV-cache mem: beam=8 fits fine on all sizes; beam=32 may OOM on d=768 with
  the 16 GiB eval GPU — will have to gate.

## Run-tag / output layout

One directory per run, parallel to the existing `{grpo, bgrpo, bgrpo_rank}`
layout, with `steps` and `init` suffixed on the tag:

```
data_storage/things_on_paper/BGRPO/runs/
    bgrpo_rank_s200_d256/
        checkpoint-5/ … checkpoint-200/
        checkpoint-final/
        eval/checkpoint-N/{greedy,beam}_predictions.txt, summary.json
    bgrpo_rank_s200_d512/
        …
```

Reasoning:
- Keeps the current 4 dirs intact so nothing breaks.
- `_s{steps}_d{init_size}` suffix keeps all 18 (phase 1+3) runs sortable.
- `plot_bgrpo_iter_curve.py` becomes `plot_bgrpo_iter_curve.py <tag-prefix>`
  so we can plot each phase independently.

## Invariants to keep across runs

1. Same 60-problem test set, same beam sweep {1..30} during eval. (Paper used
   beam=7 for eval; we already go up to 30 for compute-vs-acc trading.)
2. PAD-strip in `correctness_fn` stays on (bug fix from earlier this session).
3. Single-GPU pinning for training is NOT required — SLURM allocation picks
   one GPU anyway. `CUDA_VISIBLE_DEVICES` stays unset.
4. wandb: keep the beefed telemetry (reward stats, grad_norm, log_ratio_std,
   policy/ref logprob means) so we can diagnose any weird run.

## Decision needed before launching

- Approve Phase 1 (9 runs, ~1 hr wallclock with 3-way parallelism).
- Phase 2 / Phase 3 gated on Phase 1 reading — revisit once curves are in.
