#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00

# # # # # # Training command
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model6.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment6

# # # # # # # additional epoch command
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model6_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model6_2.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment6_additional_epochs


# # # # # # # additional epoch command2
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model6_2_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model6_3.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment6_additional_epochs2


# # # # # # # additional epoch command3
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 3 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model6_3_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model6_4.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment6_additional_epochs3


# # # # # # # additional epoch command4
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 3 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model6_4_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model6_5.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment6_additional_epochs4


# # # # # # # additional epoch command5
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 3 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model6_5_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model6_6.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment6_additional_epochs5

# # # # # # # additional epoch command6
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 3 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model6_6_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model6_7.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment6_additional_epochs6

# # # # # # # additional epoch command7
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 3 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model6_7_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model6_8.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment6_additional_epochs7

# # # # beam evaluation
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --beam_width 30 \
#    --max_test 500 \
#    --check_path ../nanogpt/check.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model6_8_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check6_beam.txt


#  # # # # # # additional training on new dataset3
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 100 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model6_8_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model6_9.pt \
#    --dataset_name dataset3_2 \
#    --exp_name experiment6_train_on_new_dataset_3

 # # # # # # additional training on new dataset4
python3 ../nanogpt/run.py inequality_finetune \
   --block_size 850 \
   --num_epochs 5 \
   --n_embd 512 \
   --n_layer 6 \
   --max_number_token 101 \
   --iteration_period 6000 \
   --lr_decay 1 \
   --shuffle 1 \
   --batch_size 100 \
   --finetune_lr 0.0006 \
   --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_valid.txt \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_test.txt \
   --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_train.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model6_9_best.pt \
   --writing_params_path ../../../data_storage/things_on_paper/model/model6_10.pt \
   --dataset_name dataset3_2 \
   --exp_name experiment6_train_on_new_dataset_4