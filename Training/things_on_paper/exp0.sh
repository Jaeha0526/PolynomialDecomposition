#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00

# # # # # # Training command
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 400 \
#    --num_epochs 15 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 3000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 256 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data0_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data0_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data0_train.txt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model0.pt \
#    --dataset_name dataset0 \
#    --exp_name experiment0


# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 400 \
#    --max_output_length 50 \
#    --n_embd 512 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data0_test.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model0_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/exp0_prediction.txt



# # # # # beam evaluation
python ../nanogpt/run.py debug_beam \
   --block_size 400 \
   --max_output_length 50 \
   --n_embd 512 \
   --n_layer 4 \
   --beam_width 10 \
   --max_test 1000 \
   --check_path ../nanogpt/check_single.m \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data0_test.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model0_best.pt \
   --outputs_path ../nanogpt/symbolic/predictions/exp0_beam.txt



# # # # # beam evaluation
python ../nanogpt/run.py debug_beam \
   --block_size 400 \
   --max_output_length 50 \
   --n_embd 512 \
   --n_layer 4 \
   --beam_width 30 \
   --max_test 10 \
   --check_path ../nanogpt/check_single.m \
   --evaluate_corpus_path ../../../dataset_storage/on_the_paper/dataset/data0_test.txt \
   --reading_params_path ../../../dataset_storage/on_the_paper/model/model0_best.pt \
   --outputs_path ../nanogpt/symbolic/predictions/exp0_beam.txt
