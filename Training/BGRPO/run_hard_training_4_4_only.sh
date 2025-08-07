#!/bin/bash

# Run BGRPO training with only 4_4 (hardest) training examples
# and validation on test_dataset_4_4

echo "Training on 4_4 examples only (degree 16 polynomials)..."

python grpo_single_variable_hard_validation.py \
    --model_name single_variable_model_best.pt \
    --reward_type rank \
    --output_dir ../../data_storage/outputs/hard_training_4_4_only \
    --dataset_path ../../data_storage/dataset/single_variable/training_dataset_4_4_filtered.txt \
    --num_generations 25 \
    --num_questions 20 \
    --num_iterations 5 \
    --lr 5e-5 \
    --beta 0.01 \
    --total_training_samples 500 \
    --adjust_rewards true \
    --val_samples 100 \
    --eval_steps 10 \
    --beam_width_eval 10 \
    --multisample_n 5 \
    --multisample_temperature 0.7 \
    --use_beam true \
    --wandb_project bgrpo-hard-4-4-only \
    --save_steps 10000