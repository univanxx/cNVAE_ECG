#!/bin/bash

export CHECKPOINT_DIR="your_saving_dir"
export DATA_DIR="your_data_dir"
export NAME="your_experiment_name"
export PORT_NUMBER=your_port_number

CUDA_VISIBLE_DEVICES=0 python3 ../conditional/train_conditional_1d.py --root $CHECKPOINT_DIR --data_dir $DATA_DIR --name $NAME \
        --num_channels_enc 12  --num_channels_dec 12 --epochs 500 --num_postprocess_cells 4 --num_preprocess_cells 4 \
        --num_latent_scales 3 --num_latent_per_group 5 --num_cell_per_cond_enc 4 --num_cell_per_cond_dec 4 \
        --num_preprocess_blocks 4 --num_postprocess_blocks 4 --num_groups_per_scale 20 \
        --batch_size 32 --num_nf 0 --master_address localhost --master_port $PORT_NUMBER \
        --ada_groups --use_se  --num_input_channels 8 --res_dist --num_mixture_dec 20 \
        --num_process_per_node 1 --arch_instance "res_bnelu" --input_size 5000 \
        --percent_epochs 5 --fast_adamax