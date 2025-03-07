#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=0-08:00:00

# # # # # # Training command
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model3.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment3

# # # # # # # additional epoch command
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 3 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model3_2.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment3_additional_epochs


# # # # # # # additional epoch command2
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_2_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model3_3.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment3_additional_epochs2


# # # # # # # additional epoch command3
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 2 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_3_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model3_4.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment3_additional_epochs3



# # # # beam evaluation
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --beam_width 30 \
#    --max_test 300 \
#    --check_path ../nanogpt/check.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_4_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check3_beam.txt


# # # # # # # # additional epoch command4
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 1 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 32 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_4_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model3_5.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment3_additional_epochs4


# # # # # beam evaluation
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --beam_width 30 \
#    --max_test 500 \
#    --check_path ../nanogpt/check.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_5_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check3_beam.txt


# # # # # # # # additional data 
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 6000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 100 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_5_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model3_add1M_1.pt \
#    --dataset_name dataset3_2 \
#    --exp_name experiment3_add1M

# # # # # # # # additional data 
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 6000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 100 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_add1M_1_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model3_add1M_2.pt \
#    --dataset_name dataset3_2 \
#    --exp_name experiment3_add1M_2


# # # # # beam evaluation
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --beam_width 30 \
#    --max_test 500 \
#    --check_path ../nanogpt/check.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_add1M_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check3_beam.txt


# # # # # # # # only increasing batch size for comparison
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 6000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 100 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_5_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model3_batch100.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment3_batch100

#    # # # # # # # only increasing batch size for comparison
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 6000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 100 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_2_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model3_add1M_1_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model3_add1M_2.pt \
#    --dataset_name dataset3_2 \
#    --exp_name experiment3_add1M_22

# # # # beam evaluation
python ../nanogpt/run.py debug_beam \
   --block_size 850 \
   --max_output_length 150 \
   --n_embd 512 \
   --n_layer 4 \
   --beam_width 30 \
   --max_test 500 \
   --check_path ../nanogpt/check.m \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model3_add1M_22_best.pt \
   --outputs_path ../nanogpt/symbolic/predictions/check3_beam_2.txt