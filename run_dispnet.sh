#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
model_name="dispnet_driving_320_512"

python -u train_dispnet.py \
--model_type='dispnet' \
--source_dataset='driving' \
--batch_size=4 \
--test_batch_size=4 \
--lr_rate=2e-5 \
--total_epochs=101 \
--save_interval=5 \
--print_freq=220 \
--checkpoint_save_path="stereogan_checkpoints/debug/${model_name}" \
--load_checkpoints 0 \
--load_from_mgpus_model 0 \
--writer=${model_name} \
--use_multi_gpu=1 \
--img_height=320 \
--img_width=512 \
--maxdisp=192 \
2>&1 | tee ./logs/train-$model_name-$now.log &

