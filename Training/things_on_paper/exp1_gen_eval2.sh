#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=0-4:00:00


python ../nanogpt/run.py inequality_evaluate4 \
    --block_size 256 \
    --max_output_length 50 \
    --n_embd 512 \
    --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/data1_test.txt \
    --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
    --outputs_path ../nanogpt/symbolic/predictions/exp1_2_predictions.txt


for name in 1280 2560 5120 10240 20480 40960
do
    for i in {1..3}
    do
        python ../nanogpt/run.py inequality_finetune \
            --block_size 256 \
            --num_epochs 1 \
            --n_embd 512 \
            --lr_decay 0 \
            --shuffle 1 \
            --batch_size 64 \
            --finetune_lr 0.0006 \
            --max_number_token 101 \
            --valid_corpus_path ../../../data_storage/things_on_paper/dataset/Gen1_valid.txt \
            --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/Gen1_test.txt \
            --finetune_corpus_path ../../../data_storage/things_on_paper/dataset/Gen1_${name}_${i}.txt \
            --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_best.pt \
            --writing_params_path ../../../data_storage/things_on_paper/model/model1_2_gen1_${name}_${i}.pt \
            --dataset_name gen1_${name}_${i} \
            --exp_name experiment1_2_gen1_${name}_${i}


        python ../nanogpt/run.py inequality_evaluate4 \
            --block_size 256 \
            --max_output_length 50 \
            --n_embd 512 \
            --evaluate_corpus_path ../../../data_storage/things_on_paper/dataset/Gen1_test.txt \
            --reading_params_path ../../../data_storage/things_on_paper/model/model1_2_gen1_${name}_${i}.pt \
            --outputs_path ../nanogpt/symbolic/predictions/exp1_2_gen1_${name}_${i}_predictions.txt
    
    done

done
