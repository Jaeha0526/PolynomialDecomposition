#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00


# # # # # # Trained by Gio, curriculum learning


python ../nanogpt/run.py debug_beam \
   --block_size 1300 \
   --max_output_length 150 \
   --n_embd 512 \
   --n_layer 4 \
   --beam_width 30 \
   --max_test 300 \
   --check_path ../nanogpt/check.m \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model9.pt \
   --outputs_path ../nanogpt/symbolic/predictions/check9_beam.txt
