#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=0-4:00:00

# # # # # Training command
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 300 \
#    --num_epochs 15 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 3000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 256 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_train1.txt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model1_2.pt \
#    --dataset_name dataset1_2_1 \
#    --exp_name experiment1_2

# # # # # # additional epoch command
# python3 ../nanogpt/run.py inequality_finetune \
#    --block_size 300 \
#    --num_epochs 15 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 6000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 256 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_train2.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M.pt \
#    --dataset_name dataset1_2_2 \
#    --exp_name experiment1_2_add1M

# python3 ../nanogpt/run.py inequality_finetune \
  #  --block_size 300 \
  #  --num_epochs 5 \
  #  --n_embd 512 \
#    --n_layer 6 \
#    --max_number_token 101 \
#    --iteration_period 6000 \
#    --lr_decay 1 \
#    --shuffle 1 \
#    --batch_size 256 \
#    --finetune_lr 0.0006 \
#    --valid_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_valid.txt \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test.txt \
#    --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_train2.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M_best.pt \
#    --writing_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M.pt \
#    --dataset_name dataset1_2_2 \
#    --exp_name experiment1_2_add1M_4_epochs

# Initial Model
# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_22.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/init_test_22_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_23.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/init_test_23_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_24.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/init_test_24_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_32.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/init_test_32_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_33.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/init_test_33_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_34.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/init_test_34_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_42.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/init_test_42_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_43.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/init_test_43_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_44.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/init_test_44_prediction.txt





# # 1M model

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_22.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/1M_test_22_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_23.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/1M_test_23_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_24.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/1M_test_24_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_32.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/1M_test_32_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_33.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/1M_test_33_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_34.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/1M_test_34_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_42.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/1M_test_42_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_43.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/1M_test_43_prediction.txt

# python ../nanogpt/run.py inequality_evaluate4 \
#    --block_size 300 \
#    --max_output_length 75 \
#    --n_embd 512 \
#    --n_layer 6 \
#    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_2_test_44.txt \
#    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_add1M_best.pt \
#    --outputs_path ../nanogpt/symbolic/predictions/1M_test_44_prediction.txt
