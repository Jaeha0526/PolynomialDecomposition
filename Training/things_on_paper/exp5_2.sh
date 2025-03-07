#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=0-23:00:00

# # # # # Training command
# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 15 \
#    --n_embd 256 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 100 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model5_256.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment5_2

# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 6 \
#    --n_embd 256 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 100 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model5_256_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model5_256.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment5_2

# python ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 4 \
#    --n_embd 256 \
#    --n_layer 4 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 100 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data3_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data3_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model5_256_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model5_256.pt \
#    --dataset_name dataset3 \
#    --exp_name experiment5_2_additional

# # # # beam evaluation
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 256 \
#    --n_layer 4 \
#    --beam_width 50 \
#    --max_test 300 \
#    --check_path ../nanogpt/check.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model5_256_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check5_256_beam.txt

# # # # # # additional 1M data
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 20 \
#    --n_embd 256 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model5_256_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model5_256_add1M.pt \
#    --dataset_name dataset3_2 \
#    --exp_name experiment5_add1M

# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 20 \
#    --n_embd 256 \
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
#    --reading_params_path ../../../data_storage/things_on_paper/model/model5_256_add1M_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model5_256_add1M.pt \
#    --dataset_name dataset3_2 \
#    --exp_name experiment5_add1M_remaining

# # # # beam evaluation
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 256 \
#    --n_layer 4 \
#    --beam_width 50 \
#    --max_test 300 \
#    --check_path ../nanogpt/check.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model5_256_add1M_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check5_256_add1M_beam.txt


# # # # beam evaluation
python ../nanogpt/run.py debug_beam \
   --block_size 850 \
   --max_output_length 150 \
   --n_embd 256 \
   --n_layer 4 \
   --beam_width 30 \
   --max_test 300 \
   --check_path ../nanogpt/check.m \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model5_256_best.pt \
   --outputs_path ../nanogpt/symbolic/predictions/check5_256_beam_30.txt

# # # # beam evaluation
python ../nanogpt/run.py debug_beam \
   --block_size 850 \
   --max_output_length 150 \
   --n_embd 256 \
   --n_layer 4 \
   --beam_width 30 \
   --max_test 300 \
   --check_path ../nanogpt/check.m \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data3_test_use_this.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model5_256_add1M_best.pt \
   --outputs_path ../nanogpt/symbolic/predictions/check5_256_add1M_beam_30.txt
