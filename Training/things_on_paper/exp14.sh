#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=0-15:00:00

# # # # # # Training command
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --n_head 4 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 200 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data6_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data6_train.txt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model14.pt \
#    --dataset_name dataset6 \
#    --exp_name experiment14

# # # # # # Training command 2
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --n_head 4 \
#    --max_number_token 101 \
#    --iteration_period 8000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 200 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data6_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data6_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model14_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model14_2.pt \
#    --dataset_name dataset6 \
#    --exp_name experiment14_additional_epochs

# # # # # # Training command 3
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 7 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --n_head 4 \
#    --max_number_token 101 \
#    --iteration_period 4000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 200 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data6_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data6_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model14_2_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model14_3.pt \
#    --dataset_name dataset6 \
#    --exp_name experiment14_additional_epochs_2

# # # # # # Training command 4
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 850 \
#    --num_epochs 10 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --n_head 4 \
#    --max_number_token 101 \
#    --iteration_period 4000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 200 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data6_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data6_train.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model14_3_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model14_4.pt \
#    --dataset_name dataset6 \
#    --exp_name experiment14_additional_epochs_3

# # # beam evaluation (ON HPC) 1
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --beam_width 30 \
#    --max_test 300 \
#    --check_path ../nanogpt/check_var3.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model14_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check14_beam.txt

# # # beam evaluation (ON HPC) 2
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --beam_width 30 \
#    --max_test 300 \
#    --check_path ../nanogpt/check_var3.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model14_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check14_beam_2.txt

# # # beam evaluation (ON HPC) 3
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --beam_width 30 \
#    --max_test 300 \
#    --check_path ../nanogpt/check_var3.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model14_3_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check14_beam_3.txt

# # # beam evaluation (ON HPC) 4
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --beam_width 50 \
#    --max_test 300 \
#    --check_path ../nanogpt/check_var3.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model14_4_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check14_beam_4.txt

# # # beam evaluation (ON HPC) 5
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --beam_width 50 \
#    --max_test 300 \
#    --check_path ../nanogpt/check_var3.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test_2.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model14_4_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check14_beam_5.txt

# # # beam evaluation (ON HPC) 6
# python ../nanogpt/run.py debug_beam \
#    --block_size 850 \
#    --max_output_length 150 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --beam_width 50 \
#    --max_test 300 \
#    --check_path ../nanogpt/check_var3.m \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test_DAG.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model14_4_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/check14_beam_DAG.txt

# # # beam evaluation (ON HPC) 7
python ../nanogpt/run.py debug_beam \
   --block_size 850 \
   --max_output_length 150 \
   --n_embd 512 \
   --n_layer 6 \
   --beam_width 15 \
   --max_test 1200 \
   --check_path ../nanogpt/check_var3.m \
   --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data6_test_DAG_large.txt \
   --reading_params_path ../../../data_storage/things_on_paper/model/model14_4_best.pt \
   --outputs_path ../nanogpt/symbolic/predictions/check14_beam_DAG_large.txt
