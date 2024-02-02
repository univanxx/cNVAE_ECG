#!/bin/bash

export CODE_DIR=$1
export SAVE_DIR=$2
export LOG_PATH=$3
export LOG_NAME=$4
export NUM_CHANNELS=$5
export CLASS_TYPE=$6

cd $CODE_DIR

for temp_i in 0.7 0.8 0.9
do
    python evaluate_conditional_1d.py --checkpoint $LOG_PATH/$LOG_NAME/checkpoint.pt \
        --master_address localhost --master_port 11237 --eval_mode=sample \
        --temp=$temp_i --save ${SAVE_DIR}${LOG_NAME} --num_input_channels $NUM_CHANNELS --class_type $CLASS_TYPE --readjust_bn --num_iters 100
done