#!/bin/bash


export CHECKPOINT_DIR=$1
export CODE_DIR=$2
export DATA_DIR=$3
export "CLASS_NAME=$4"
export NAME=$5


cd $CODE_DIR


# CUDA_VISIBLE_DEVICES=0 python train.py --data $DATA_DIR/celeba64_lmdb --root $CHECKPOINT_DIR --save $EXPR_ID --dataset celeba_64 \
#         --num_channels_enc 64 --num_channels_dec 64 --epochs 90 --num_postprocess_cells 2 --num_preprocess_cells 2 \
#         --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
#         --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 20 \
#         --batch_size 16 --num_nf 1 --ada_groups --num_process_per_node 8 --use_se --res_dist --fast_adamax --master_port 11235 --imtype real --num_input_channels 1

# USUAL PIPELINE 
PATH+=:/home/jovyan/isviridov/gans/gan_venv/bin/
CUDA_VISIBLE_DEVICES=1 python3 train_conditional_1d.py --root $CHECKPOINT_DIR --data_dir $DATA_DIR --name $NAME \
        --num_channels_enc 12  --num_channels_dec 12 --epochs 500 --num_postprocess_cells 4 --num_preprocess_cells 4 \
        --num_latent_scales 3 --num_latent_per_group 5 --num_cell_per_cond_enc 4 --num_cell_per_cond_dec 4 \
        --num_preprocess_blocks 4 --num_postprocess_blocks 4 --num_groups_per_scale 20 \
        --batch_size 64 --num_nf 0 --master_address localhost --master_port 1213 \
        --ada_groups --use_se  --num_input_channels 8 --res_dist --class_name "$CLASS_NAME" --num_mixture_dec 20 \
        --num_process_per_node 1 --arch_instance "res_bnelu" --input_size 2560 --dataset "non_compete" \
        --percent_epochs 5 --fold_idx 3 --sample --cont_training #--filter  #--focal # #--cont_training #--weight_decay_norm 0.01 #   --squared

# CUDA_VISIBLE_DEVICES=0 python train_conditional.py --root $CHECKPOINT_DIR \
#         --num_channels_enc 16 --num_channels_dec 16 --epochs 500 --num_postprocess_cells 2 --num_preprocess_cells 2 \
#         --num_latent_scales 3 --num_latent_per_group 5 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
#         --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 10 \
#         --batch_size 64 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax \
#         --class_name "$CLASS_NAME" --num_mixture_dec 10 --num_input_channels 2 --master_address localhost --master_port 1213 --easy 


# CUDA_VISIBLE_DEVICES=0 python train_conditional.py --root $CHECKPOINT_DIR \
#         --num_channels_enc 64 --num_channels_dec 64 --epochs 90 --num_postprocess_cells 2 --num_preprocess_cells 2 \
#         --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
#         --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 20 \
#         --batch_size 16 --num_nf 0 --ada_groups  --num_mixture_dec 10 --num_process_per_node 1 --use_se --res_dist \
#         --fast_adamax --master_address localhost --master_port 1213 --easy --squared --class_name "$CLASS_NAME"





# ./training_conditional.sh /media/ssd-3t/isviridov/mdetr_work/gans/nvae/nvae_debug_logs \ 
# /media/ssd-3t/isviridov/mdetr_work/gans/nvae/my_NVAE/NVAE "myocard" --fold_idx 2

# --easy --cont_training #--arch_instance "res_bnelu"
# --data $DATA_DIR/celeba64_lmdb --save $EXPR_ID --dataset celeba_64 --ecg_channel 0 --num_input_channels 3


# ./training.sh binary_generation \
# /gim/lv02/isviridov/code/gan_ecg/nvae_debug_logs \
# /gim/lv02/isviridov/code/gan_ecg/NVAE
# /media/ssd-3t/isviridov/mdetr_work/gans/data/celeba_org \





# CUDA_VISIBLE_DEVICES=0 python train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset celeba_64 --batch_size 200 \
#         --epochs 400 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
#         --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 10 --num_preprocess_blocks 2 \
#         --num_postprocess_blocks 2 --weight_decay_norm 1e-1 --num_channels_enc 8 --num_channels_dec 8 --num_nf 0 \
#         --ada_groups --use_se --res_dist --fast_adamax --master_port 11235 --imtype real --num_input_channels 3
        
        
        
# CUDA_VISIBLE_DEVICES=1 python train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset celeba_64 \
#         --num_channels_enc 32 --num_channels_dec 32 --epochs 400 --num_postprocess_cells 3 --num_preprocess_cells 3 \
#         --num_latent_scales 2 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
#         --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-1 --num_groups_per_scale 10 \
#         --batch_size 200 --num_nf 0 --master_address localhost --master_port 11235 --use_se --imtype real --ada_groups --use_se --res_dist --fast_adamax


# args = Namespace(ada_groups=True, arch_instance='res_mbconv', batch_size=64, cont_training=False, data='/media/ssd-3t/isviridov/mdetr_work/gans/data/celeba_org/celeba64_lmdb', dataset='celeba_64', detect_anomalies=False, distributed=True, epochs=500, fast_adamax=True, global_rank=0, imtype='all', input_size=64, kl_anneal_portion=0.3, kl_const_coeff=0.0001, kl_const_portion=0.0001, learning_rate=0.01, learning_rate_min=0.0001, local_rank=0, master_address='localhost', master_port='11237', min_groups_per_scale=1, node_rank=0, num_cell_per_cond_dec=1, num_cell_per_cond_enc=1, num_channels_dec=8, num_channels_enc=8, num_groups_per_scale=20, num_input_channels=3, num_latent_per_group=5, num_latent_scales=3, num_mixture_dec=10, num_nf=1, num_postprocess_blocks=1, num_postprocess_cells=2, num_preprocess_blocks=1, num_preprocess_cells=2, num_proc_node=1, num_process_per_node=1, num_total_iter=152500, num_x_bits=8, res_dist=True, root='/media/ssd-3t/isviridov/mdetr_work/gans/nvae/nvae_debug_logs', save='/media/ssd-3t/isviridov/mdetr_work/gans/nvae/nvae_debug_logs/eval-ptbxl_big_together', seed=1, use_se=True, warmup_epochs=5, weight_decay=0.0003, weight_decay_norm=0.1, weight_decay_norm_anneal=False, weight_decay_norm_init=10.0)