#!/bin/bash
# Launch BGRPO training with beam rank score metric

# Navigate to the BGRPO directory
cd /workspace/PolynomialDecomposition/Training/BGRPO

# Run with enhanced metrics
python grpo_single_variable_beam_rank_score.py \
    --model_dir /workspace/PolynomialDecomposition/data_storage/model/ \
    --config_path /workspace/PolynomialDecomposition/data_storage/model/model_configurations/model_configuration.json \
    --train_dataset ../../data_storage/dataset/single_variable/training_dataset_4_4_beam25_500samples.txt \
    --val_dataset ../../data_storage/dataset/single_variable/test_dataset_4_4.txt \
    --output_dir ./checkpoints_beam_rank_score/ \
    --val_samples 100 \
    --use_beam true \
    --num_generations 15 \
    --lr 1e-4 \
    --beta 0.02 \
    --training_samples 400 \
    --project bgrpo-beam-rank-score \
    --run_name bgrpo_beam_rank_score_exp