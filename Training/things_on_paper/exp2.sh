#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00

# # # # # # Training command
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 256 \
#    --num_epochs 15 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 3000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 256 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data1_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data2_train.txt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model2.pt \
#    --dataset_name dataset2 \
#    --exp_name experiment2

# # # # # # additional epoch command
python3 ../nanogpt/run.py inequality_finetune \
   --block_size 256 \
   --num_epochs 15 \
   --n_embd 512 \
   --n_layer 4 \
   --max_number_token 101 \
   --iteration_period 6000 \
   --lr_decay 1 \
   --shuffle 1 \
   --batch_size 256 \
   --finetune_lr 0.0006 \
   --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data1_valid.txt \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_test.txt \
   --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data2_train.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model2_best.pt \
   --writing_params_path ../../../data_storage/things_on_paper/model/model2_2.pt \
   --dataset_name dataset2 \
   --exp_name experiment2_additional_epochs

# # # # # # additional epoch command
python3 ../nanogpt/run.py inequality_finetune \
   --block_size 256 \
   --num_epochs 15 \
   --n_embd 512 \
   --n_layer 4 \
   --max_number_token 101 \
   --iteration_period 6000 \
   --lr_decay 1 \
   --shuffle 1 \
   --batch_size 256 \
   --finetune_lr 0.0006 \
   --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data1_valid.txt \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_test.txt \
   --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data2_train.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model2_2_best.pt \
   --writing_params_path ../../../data_storage/things_on_paper/model/model2_3.pt \
   --dataset_name dataset2 \
   --exp_name experiment2_additional_epochs2


