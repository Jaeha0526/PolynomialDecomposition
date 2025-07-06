EXP_DIR=../../data_storage/outputs
OUTPUT_DIR=../../data_storage/model/BGRPO
CHECKPOINTS=(
    10
    20
    30
    40
    50
    60
    70
    80
    90
    100
)

EXP_NAME=_BGRPO
for CHECKPOINT in ${CHECKPOINTS[@]}; do
    python convert_safetensors_to_pt.py \
        --input_dir=${EXP_DIR}/${EXP_NAME}/checkpoint-${CHECKPOINT} \
            --output_dir=${OUTPUT_DIR}/${EXP_NAME} \
                --output_model_name=pytorch_model_${EXP_NAME}_${CHECKPOINT}
done