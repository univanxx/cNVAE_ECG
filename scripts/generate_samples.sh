#!/bin/bash

export PORT_NUMBER=your_port_number

CUDA_VISIBLE_DEVICES=0 python ../conditional/generate_signals.py --checkpoint "your_checkpoint_path" \
    --labels_path "your_data_path/labels.npy" \
    --task_type "imbalance" \
    --save_path "your_save_path" \
    --master_address localhost --master_port $PORT_NUMBER --eval_mode=sample --num_input_channels 8 --readjust_bn --num_iters 100 --batch_size 32