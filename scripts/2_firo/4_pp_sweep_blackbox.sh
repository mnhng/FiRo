#!/bin/bash
source activate firo

TASKS=(mnli mrpc qnli qqp rte sst2)
MAX_NUM_WORDS=(1 2 3 4 5 6 7)

OUT_DIR=output

for TASK_NAME in ${TASKS[@]}; do
for NUM_WORD in ${MAX_NUM_WORDS[@]}; do 
    python glue_blackbox_evaluate.py --task_name $TASK_NAME --do_lower_case \
        --model_name_or_path $OUT_DIR/ft_base_firo/$TASK_NAME \
        --output_dir $OUT_DIR/black_firo_pp \
        --recoverer firo --rec_path data/firo/chkpt.pt \
        --max_num_words $NUM_WORD \
        --beam_width 5
done
done
