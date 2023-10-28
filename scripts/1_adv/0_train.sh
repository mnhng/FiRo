#!/bin/bash
source activate firo

TASKS=(mnli mrpc qnli qqp rte sst2)

OUT_DIR=output

for TASK_NAME in ${TASKS[@]}; do
    python glue_adv_finetune.py --task_name $TASK_NAME --do_lower_case --do_train \
        --model_name_or_path $OUT_DIR/$TASK_NAME \
        --output_dir $OUT_DIR/adversarial/$TASK_NAME --overwrite_output_dir \
        --recoverer identity
done
