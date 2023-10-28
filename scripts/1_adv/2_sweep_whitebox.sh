#!/bin/bash
source activate firo

TASKS=(mnli mrpc qnli qqp rte sst2)
MAX_NUM_WORDS=(1 2 3 4 5 6 7)

OUT_DIR=output

for TASK_NAME in ${TASKS[@]}; do
for NUM_WORD in ${MAX_NUM_WORDS[@]}; do 
    python glue_adv_evaluate.py --task_name $TASK_NAME --do_lower_case \
        --probe_model $OUT_DIR/adversarial/$TASK_NAME \
        --adv_model $OUT_DIR/adversarial/$TASK_NAME \
        --output_dir $OUT_DIR/white_adv \
        --max_num_words $NUM_WORD \
        --beam_width 5
done
done
