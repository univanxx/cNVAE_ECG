#!/bin/bash

for fold_i in 0 1 2 3 4
do
    python run_hypo_test.py -model_name "" -data_path "" \
    -model_type "nvae" -generated_path "" -fold $fold_i --seed 23 \
    --res_path "" --device 0
done