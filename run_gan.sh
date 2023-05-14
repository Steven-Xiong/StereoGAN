#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
model_name="gan_driving_withflow"

python -u train_gan.py \
--model_type='GAN' \
--source_dataset='driving' \
--batch_size=4 \
--lr_gan=2e-4 \
--train_ratio_gan=3 \
--total_epochs=11 \
--save_interval=5 \
--print_freq=220 \
--checkpoint_save_path="stereogan_checkpoints/${model_name}" \
--load_checkpoints 0 \
--load_from_mgpus_model 0 \
--writer=${model_name} \
--use_multi_gpu=2 \
--img_height=256 \
--img_width=512 \
--maxdisp=192 \
--lambda_corr=0 \
--lambda_cycle=10 \
--lambda_id=5 \
--lambda_ms=0.1 \
--lambda_warp_inv=5 \
--lambda_disp_warp_inv=5 \
--lambda_warp=5 \
--lambda_disp_warp=5 \
--cosine_similarity=1 \
--perceptual=1 \
--smooth_loss=0 \
--unimatch_stereo 0 \
--lambda_flow_warp 5 \
--lambda_flow_warp_inv 5 \
--lambda_flow_warpx 5 \
--lambda_flow_warpx_inv 5 \
--debug 0 \
2>&1 | tee ./logs/train-$model_name-$now.log &

