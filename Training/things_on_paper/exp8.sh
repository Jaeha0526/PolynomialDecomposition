#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu
#SBATCH --time=0-10:01:00


# # # # # # Training command
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 1024 \
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
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment8

# # # # # # # additional epoch command
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_2.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment8_additional_epochs


# # # # # # # additional epoch command2
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_2_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_3.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment8_additional_epochs2


# # # # # # additional epoch command3
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_3_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_4.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment8_additional_epochs3



# # # # # # additional epoch command4
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_4_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_5.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment8_additional_epochs4


# # # # # # additional epoch command5
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_5_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_6.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment8_additional_epochs5


# # # # # # additional epoch command5
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_6_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_7.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment8_additional_epochs6

# # # # # additional epoch command6
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_7_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_8.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment8_additional_epochs7



# # # # # additional epoch command7
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_8_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_9.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment8_additional_epochs8



# # # # beam evaluation
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 1024 \
#    --n_layer 6 \
#    --beam_width 30 \
#    --max_test 300 \
#    --check_path ../nanogpt/check.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_9_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check8_beam.txt


# # # # # # additional training on new dataset
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_9_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_10.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment8_train_on_new_dataset_1


#    # # # # # additional training on new dataset2
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_10_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_11.pt \
#    --dataset_name dataset3_2 \
#    --exp_name experiment8_train_on_new_dataset_2

# # # # # # additional training on new dataset3
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 5 \
#    --n_embd 1024 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model8_11_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model8_12.pt \
#    --dataset_name dataset3_2 \
#    --exp_name experiment8_train_on_new_dataset_3


# # # beam evaluation
python3 ../nanogpt/run.py debug_beam \
   --block_size 850 \
   --max_output_length 150 \
   --n_embd 1024 \
   --n_layer 6 \
   --beam_width 30 \
   --max_test 300 \
   --check_path ../nanogpt/check.m \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model8_12_best.pt \
   --outputs_path ../nanogpt/symbolic/predictions/check8_after_additional1M_beam.txt