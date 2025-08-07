#!/bin/bash
# Launch BGRPO training with beam rank score metric

cd /workspace/PolynomialDecomposition/Training/BGRPO

python grpo_single_variable_with_beam_score.py \
    --num_generations 15 \
    --use_beam \
    --lr 1e-4 \
    --beta 0.02 \
    --total_training_samples 400 \
    --dataset_path ../../data_storage/dataset/single_variable/training_dataset_4_4_beam25_500samples.txt \
    --output_dir ./checkpoints_beam_score/ \
    --eval_steps 10 \
    --beam_width_eval 25 \
    --val_samples 100 \
    --multisample_n 10 \
    --multisample_temperature 0.7 \
    --plot_interval 10