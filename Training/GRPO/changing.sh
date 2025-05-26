EXP_DIR=../../outputs
OUTPUT_DIR=../../data_storage/model/BGRPO
CHECKPOINTS=(
    20
    40
    60
    80
    100
    120
    140
    160
    180
    200
)

EXP_NAME=single_variable
for CHECKPOINT in ${CHECKPOINTS[@]}; do
    python convert_safetensors_to_pt.py \
        --input_dir=${EXP_DIR}/${EXP_NAME}/checkpoint-${CHECKPOINT} \
            --output_dir=${OUTPUT_DIR}/${EXP_NAME} \
                --output_model_name=pytorch_model_${EXP_NAME}_${CHECKPOINT}
done