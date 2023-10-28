#!/bin/bash
source activate firo
export WANDB_DISABLED=true

OUT_DIR="output/ner"

python ner_exp.py \
    --model_name_or_path $OUT_DIR/base --dataset_name conll2003 \
    --adv_model_path $OUT_DIR/ft_base \
    --output_dir $OUT_DIR/sweep_ft_firo \
    --do_eval \
    --recoverer firo --rec_path data/firo/chkpt.pt

MAX_NUM_WORDS=(1 2 3 4 5 6 7)
for I in ${MAX_NUM_WORDS[@]}; do 
    python ner_exp.py \
        --model_name_or_path $OUT_DIR/base --dataset_name conll2003 \
        --adv_model_path $OUT_DIR/ft_base \
        --output_dir $OUT_DIR/sweep_ft_firo \
        --do_eval --adversarial \
        --max_num_words $I \
        --recoverer firo --rec_path data/firo/chkpt.pt
done
