# BGRPO Training Scripts

This directory contains scripts for training polynomial decomposition models using BGRPO (Beam Group Relative Policy Optimization) and GRPO (Group Relative Policy Optimization) with TRL.

## Core Training Scripts

### Main Training Scripts
- **`grpo_single_variable_with_beam_score.py`** - Enhanced BGRPO/GRPO training with beam rank score metric and pass@10 evaluation
  - Adds beam rank score: exp(-(rank-1)/10) for correct answers within beam width 10
  - Dual beam width evaluation (10 for scoring, 25 for full accuracy)
  - Comprehensive plotting system with 3 subplots
  - Full WandB integration

- **`grpo_single_variable_hard_validation.py`** - BGRPO/GRPO training with hard validation on test_dataset_4_4
  - Focuses on hardest test cases (degree 16 polynomials)
  - Real-time validation plotting
  - Used in WandB sweeps for hyperparameter optimization

- **`grpo_single_variable_with_validation.py`** - Basic BGRPO/GRPO training with validation
  - Standard validation evaluation
  - Checkpoint saving at best accuracy
  - Real-time plotting of metrics

- **`grpo_single_variable_beam_rank_score.py`** - Initial attempt at beam rank score (superseded by with_beam_score.py)

## Dataset Preparation Scripts

- **`filter_training_dataset.py`** - Filters training dataset for specific polynomial degrees (e.g., 4_4 examples)
- **`filter_dataset_beam_corrected.py`** - Creates filtered dataset where answers exist within top-25 beam search results
  - Ensures at least one positive signal during BGRPO training
  - Properly handles polynomial comparison (ignoring constant factors)

## Evaluation Scripts

- **`evaluate_test_datasets.py`** - Evaluates trained models on all test datasets with beam search
- **`plot_test_results.py`** - Creates visualization of test results across different polynomial degrees

## WandB Sweep Scripts

- **`run_bgrpo_vs_grpo_sweep.py`** - Main sweep launcher comparing BGRPO vs GRPO
- **`run_bgrpo_vs_grpo_sweep_single_gpu.py`** - Single GPU version to avoid multi-GPU issues
- **`run_bgrpo_vs_grpo_sweep_multi_agent.py`** - Multi-agent version for parallel sweep execution
- **`run_wandb_sweep_4_4.py`** - Specific sweep for 4_4 dataset training

## Analysis Scripts

- **`analyze_sweep_results.py`** - Analyzes WandB sweep results (requires API access)
- **`analyze_sweep_offline.py`** - Offline analysis of exported sweep results
- **`analyze_wandb_sweep.py`** - Alternative analysis script

## Configuration Files

- **`wandb_sweep_4_4_final.yaml`** - Final sweep configuration for BGRPO vs GRPO comparison
- **`wandb_sweep_beam_score.yaml`** - Sweep configuration for beam rank score experiments

## Launch Scripts

- **`launch_bgrpo_with_beam_score.sh`** - Launch script for beam rank score training
- **`launch_bgrpo_beam_rank_score.sh`** - Alternative launch script
- **`launch_single_gpu_sweep.sh`** - Launch sweep on single GPU
- **`run_hard_training_4_4_only.sh`** - Train on 4_4 examples only
- **`run_hard_training_mixed.sh`** - Train on mixed dataset (50% hard, 50% regular)
- **`run_hard_validation_training.sh`** - Launch hard validation training

## Documentation

- **`BGRPO_TRL_Trick_Explanation.md`** - Explains how BGRPO hijacks TRL's batching mechanism
  - Key insight: beam_width = len(input_ids) in generate()
  - Details on single GPU restriction for proper beam width

- **`KV_Cache_Guide.md`** - Complete guide to using KV-cache optimization with BGRPO
  - 3x speedup when combined with Flash Attention
  - Essential for efficient beam search and multisample evaluation
  - Simple to enable: just add `use_kvcache=True`

## Utility Scripts

- **`convert_safetensors_to_pt.py`** - Converts model checkpoints from safetensors to PyTorch format
- **`verification_summary.txt`** - Summary of beam search verification results

## Performance Optimizations

### KV-Cache (Key-Value Cache)
KV-cache provides significant speedup for autoregressive generation by caching key-value projections:
- **3x overall speedup** when combined with Flash Attention
- **Especially beneficial for beam search** - BGRPO's core operation
- **Easy to enable** - just add `use_kvcache=True` when loading models

To enable KV-cache in your training scripts:
```python
model, tokenizer = load_model_and_tokenizer(
    config_path=config_path,
    model_dir_path=model_dir_path,
    device=device,
    wrap_for_grpo=True,
    model_name=args.model_name,
    use_kvcache=True  # ← Add this line
)
```

See `KV_Cache_Guide.md` for detailed documentation.

## Usage Examples

### Basic BGRPO Training with Beam Rank Score
```bash
python grpo_single_variable_with_beam_score.py \
  --model_name single_variable_model_best.pt \
  --dataset_path ../../data_storage/dataset/single_variable/training_dataset_4_4_filtered.txt \
  --use_beam \
  --beam_width 25 \
  --lr 0.0001 \
  --beta 0.02 \
  --output_dir outputs/bgrpo_beam_score
```

### Run WandB Sweep
```bash
python run_bgrpo_vs_grpo_sweep.py --entity your-wandb-entity --agents 3
```

### Evaluate Model on Test Datasets
```bash
python evaluate_test_datasets.py \
  --checkpoint_path outputs/best_model/checkpoint-100 \
  --output_dir test_results
```

## Key Hyperparameters

- **use_beam**: true for BGRPO, false for GRPO
- **beam_width**: Typically 25 for training
- **lr**: Learning rate (1e-5 to 2e-4)
- **beta**: KL penalty coefficient (0.01 to 0.05)
- **num_generations**: Number of generation rounds (10-25)
- **num_questions**: Questions per iteration (8-20)
- **multisample_n**: Number of samples for pass@k evaluation (10)
- **beam_width_eval**: Beam width for evaluation metrics (10)