#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=0-25:00:00

# # # # beam evaluation
python ../nanogpt/run.py debug_beam \
   --block_size 850 \
   --max_output_length 150 \
   --n_embd 512 \
   --n_layer 4 \
   --beam_width 30 \
   --max_test 700 \
   --check_path ../nanogpt/check.m \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_700.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model3_5_best.pt \
   --outputs_path ../nanogpt/symbolic/predictions/model3_5_best_beam_30.txt
