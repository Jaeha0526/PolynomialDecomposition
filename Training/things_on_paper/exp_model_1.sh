#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=0-15:00:00

# # # beam evaluation (ON HPC) 7
python ../nanogpt/run.py debug_beam \
   --block_size 256 \
   --max_output_length 150 \
   --n_embd 512 \
   --n_layer 4 \
   --beam_width 15 \
   --max_test 1200 \
   --check_path ../nanogpt/check_single.m \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_test_DAG.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
   --outputs_path ../nanogpt/symbolic/predictions/model1_beam.txt
