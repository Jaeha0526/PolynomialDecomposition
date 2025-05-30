#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00


# # # # # # Training command
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 3 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --n_head 16 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../storage/thingsonpaper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../storage/thingsonpaper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../storage/thingsonpaper/dataset/data3_train.txt \
#    --writing_params_path ../../../storage/thingsonpaper/model/model10.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment10


# # # # # # Training command2
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 3 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --n_head 16 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 64 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../storage/thingsonpaper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../storage/thingsonpaper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../storage/thingsonpaper/dataset/data3_train.txt \
#    --reading_params_path ../../../storage/thingsonpaper/model/model10_best.pt \
#    --writing_params_path ../../../storage/thingsonpaper/model/model10_2.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment10_additional_epoch1


# # # # # # Training command3
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --n_head 16 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 64 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../storage/thingsonpaper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../storage/thingsonpaper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../storage/thingsonpaper/dataset/data3_train.txt \
#    --reading_params_path ../../../storage/thingsonpaper/model/model10_2_best.pt \
#    --writing_params_path ../../../storage/thingsonpaper/model/model10_3.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment10_additional_epoch2


# # # # # # Training command4
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 10 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --n_head 16 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 64 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../storage/thingsonpaper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../storage/thingsonpaper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../storage/thingsonpaper/dataset/data3_train.txt \
#    --reading_params_path ../../../storage/thingsonpaper/model/model10_3_best.pt \
#    --writing_params_path ../../../storage/thingsonpaper/model/model10_4.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment10_additional_epoch3


# # # # # # Training command4 / comes to the hpc
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 10 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --n_head 16 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 64 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model10_4_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model10_5.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment10_additional_epoch4


# # # # # # Training command5 / comes to the hpc
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 10 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --n_head 16 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model10_5_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model10_6.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment10_additional_epoch5



# # # # beam evaluation
python ../nanogpt/run.py debug_beam \
   --block_size 850 \
   --max_output_length 150 \
   --n_embd 512 \
   --n_layer 6 \
   --n_head 16 \
   --beam_width 30 \
   --max_test 300 \
   --check_path ../nanogpt/check.m \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model10_6_best.pt \
   --outputs_path ../nanogpt/symbolic/predictions/check10_beam.txt