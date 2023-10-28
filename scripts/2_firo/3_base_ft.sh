#!/bin/bash
source activate firo

TASKS=(mnli mrpc qnli qqp rte sst2)

OUT_DIR=output

for TASK_NAME in ${TASKS[@]}; do
    python glue_train.py --task_name $TASK_NAME --do_lower_case \
        --output_dir $OUT_DIR/ft_base_firo/$TASK_NAME --overwrite_output_dir \
        --recoverer firo --rec_path data/firo/chkpt.pt
done
