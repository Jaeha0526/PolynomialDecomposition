# # # # # Initial training command
python3 ../nanogpt/run.py inequality_finetune \
   --block_size 256 \
   --num_epochs 5 \
   --n_embd 256 \
   --n_layer 4 \
   --max_number_token 101 \
   --iteration_period 3000 \
   --lr_decay 1 \
   --shuffle 1 \
   --batch_size 256 \
   --finetune_lr 0.0006 \
   --valid_corpus_path ../../data_storage/dataset/example_ON/ON_data_valid.txt \
   --evaluate_corpus_path ../../data_storage/dataset/example_ON/ON_data_test.txt \
   --finetune_corpus_path ../../data_storage/dataset/example_ON/ON_data_train.txt \
   --writing_params_path ../../data_storage/model/example_ON.pt \
   --dataset_name example_ON_dataset \
   --exp_name example_ON


# # # # # # finetuning or training additional epoch command
python3 ../nanogpt/run.py inequality_finetune \
   --block_size 256 \
   --num_epochs 3 \
   --n_embd 256 \
   --n_layer 4 \
   --max_number_token 101 \
   --iteration_period 6000 \
   --lr_decay 1 \
   --shuffle 1 \
   --batch_size 256 \
   --finetune_lr 0.0006 \
   --valid_corpus_path ../../data_storage/dataset/example_ON/ON_data_valid.txt \
   --evaluate_corpus_path ../../data_storage/dataset/example_ON/ON_data_test.txt \
   --finetune_corpus_path ../../data_storage/dataset/example_ON/ON_data_train.txt \
   --reading_params_path ../../data_storage/model/example_ON_best.pt \
   --writing_params_path ../../data_storage/model/example_ON_2.pt \
   --dataset_name example_ON_dataset \
   --exp_name example_ON_2


# # # # # # # evaluate the model with greedy search
python ../nanogpt/run.py inequality_evaluate4 \
   --block_size 256 \
   --max_output_length 50 \
   --n_embd 512 \
   --n_layer 4 \
   --evaluate_corpus_path ../../data_storage/dataset/example_ON/ON_data_test.txt \
   --reading_params_path ../../data_storage/model/example_ON_2_best.pt \
   --outputs_path ../../data_storage/dataset/example_ON/example_ON_prediction.txt


# # # # evaluate the model with beam search
python ../nanogpt/run.py debug_beam \
   --block_size 256 \
   --max_output_length 50 \
   --n_embd 256 \
   --n_layer 4 \
   --beam_width 10 \
   --max_test 100 \
   --check_path ../nanogpt/check.m \
   --evaluate_corpus_path ../../data_storage/dataset/example_ON/ON_data_test.txt \
   --reading_params_path ../../data_storage/model/example_ON_2_best.pt \
   --outputs_path ../../data_storage/dataset/example_ON/example_ON_beam_search.txt


