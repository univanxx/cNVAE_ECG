#!/bin/bash

for task_type in addition imbalance
do
    for model_type in wg* p2p nvae
    do 
        python enrich_train.py -origin_data_path "your_path" \
                                -generated_data_path "your_path" \
                                -model_type $model_type  -task_type $task_type
    done
done