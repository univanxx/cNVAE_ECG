#!/bin/bash


export CHECKPOINT_DIR=$1
export CODE_DIR=$2
export DATA_DIR=$3
export "CLASS_NAME=$4"
export NAME=$5


cd $CODE_DIR

CUDA_VISIBLE_DEVICES=1,2 python3 train_conditional_1d.py --root $CHECKPOINT_DIR --data_dir $DATA_DIR --name $NAME \
        --num_channels_enc 12  --num_channels_dec 12 --epochs 500 --num_postprocess_cells 4 --num_preprocess_cells 4 \
        --num_latent_scales 3 --num_latent_per_group 5 --num_cell_per_cond_enc 4 --num_cell_per_cond_dec 4 \
        --num_preprocess_blocks 4 --num_postprocess_blocks 4 --num_groups_per_scale 20 \
        --batch_size 64 --num_nf 0 --master_address localhost --master_port 1213 \
        --ada_groups --use_se  --num_input_channels 8 --res_dist --class_name "$CLASS_NAME" --num_mixture_dec 20 \
        --num_process_per_node 2 --arch_instance "res_bnelu" --input_size 2560 \
        --percent_epochs 5 --fold_idx 0 --sample