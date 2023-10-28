#!/bin/bash
source activate firo
export WANDB_DISABLED=true

OUT_DIR="output/ner"

python ner_exp.py \
    --model_name_or_path $OUT_DIR/base --dataset_name conll2003 \
    --output_dir $OUT_DIR/sweep_identity \
    --do_eval

MAX_NUM_WORDS=(1 2 3 4 5 6 7)
for I in ${MAX_NUM_WORDS[@]}; do 
    python ner_exp.py \
        --model_name_or_path $OUT_DIR/base --dataset_name conll2003 \
        --output_dir $OUT_DIR/sweep_identity \
        --do_eval --adversarial --max_num_words $I
done
