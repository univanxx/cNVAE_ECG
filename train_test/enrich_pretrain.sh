#!/bin/bash

for dataset_name in ningbo georgia
do
    for model_type in wg* p2p nvae
    do 
        python enrich_pretrain.py -origin_data_path "your_path" \
                                -generated_data_path "your_path" \
                                -model_type $model_type  -dataset_name $dataset_name
    done
done