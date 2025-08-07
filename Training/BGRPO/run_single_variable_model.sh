#!/bin/bash

# Create directory for outputs if it doesn't exist
mkdir -p ../../data_storage/model/model_configurations

# Set GPU ID - change this as needed
export CUDA_VISIBLE_DEVICES=0

MODEL_FILE="single_variable_model_best.pt"
CONFIG_FILE="model_configuration.json"

# # First run with simple reward
# echo "Starting single variable model with simple reward on GPU ${CUDA_VISIBLE_DEVICES}"
# python grpo_ablation.py \
#   --model_name ${MODEL_FILE} \
#   --reward_type simple \
#   --output_dir ../../data_storage/outputs/_BGRPO \
#   --config_name ${CONFIG_FILE} \
#   --dataset_path ../../data_storage/dataset/single_variable/training_dataset.txt \
#   --disable_wandb \
#   --lr 1e-5 \
#   --beta 0.01 \
#   --total_training_samples 120 \
#   --num_generations 25 \
#   --num_questions 8 \
#   --num_iterations 5 \
#   --save_steps 10

# Then run with rank reward
echo "Starting single variable model with rank reward on GPU ${CUDA_VISIBLE_DEVICES}"
python grpo_ablation.py \
  --model_name ${MODEL_FILE} \
  --reward_type rank \
  --output_dir ../../data_storage/outputs/_BGRPO \
  --config_name ${CONFIG_FILE} \
  --dataset_path ../../data_storage/dataset/single_variable/training_dataset.txt \
  --disable_wandb \
  --lr 1e-5 \
  --beta 0.01 \
  --total_training_samples 400 \
  --num_generations 10 \
  --num_questions 20 \
  --num_iterations 5 \
  --save_steps 10

echo "single variable model experiments completed!"