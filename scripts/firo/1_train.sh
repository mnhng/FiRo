#!/bin/bash
source activate firo

python train_preproc.py --augment -e2 --local 1 --out_path data/firo
