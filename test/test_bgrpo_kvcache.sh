#!/bin/bash
# Quick BGRPO test with KV-cache

export CUDA_VISIBLE_DEVICES=0

MODEL_FILE="single_variable_model_best.pt"
CONFIG_FILE="model_configuration.json"

echo "Testing BGRPO with KV-cache model"
python grpo_ablation.py   --model_name ${MODEL_FILE}   --reward_type rank   --output_dir ../../test/bgrpo_kvcache_test   --config_name ${CONFIG_FILE}   --dataset_path ../../data_storage/dataset/single_variable/training_dataset.txt   --disable_wandb   --lr 1e-5   --beta 0.01   --total_training_samples 20   --num_generations 5   --num_questions 4   --num_iterations 2   --save_steps 1

echo "BGRPO KV-cache test completed!"
