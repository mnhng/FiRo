#!/bin/bash
source activate firo
export WANDB_DISABLED=true

OUT_DIR="output/ner"

python ner_exp.py --dataset_name conll2003 --do_train \
    --model_name_or_path bert-base-uncased \
    --output_dir $OUT_DIR/base \
    --recoverer identity
