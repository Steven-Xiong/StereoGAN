#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
model_name="VKITTI2_rebuttle"

python -u train_rebuttle.py \
--model_type='LEA' \
--source_dataset='VKITTI2' \
--batch_size=2 \
--test_batch_size=2 \
--lr_rate=5e-5 \
--lr_gan=1e-4 \
--train_ratio_gan=3 \
--total_epochs=101 \
--save_interval=5 \
--print_freq=5 \
--checkpoint_save_path="stereogan_checkpoints/${model_name}" \
--load_checkpoints 0 \
--load_from_mgpus_model 0 \
--writer=${model_name} \
--use_multi_gpu=2 \
--img_height=288 \
--img_width=576 \
--maxdisp=192 \
--lambda_corr=1 \
--lambda_cycle=10 \
--lambda_id=5 \
--lambda_ms=0.1 \
--lambda_warp_inv=5 \
--lambda_disp_warp_inv=5 \
--lambda_warp=0 \
--lambda_disp_warp=0 \
--cosine_similarity=1 \
--perceptual=1 \
--smooth_loss=0 \
--left_right_consistency=0 \
--flow=1 \
--num_scales 2 \
--upsample_factor 4 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--lr_flow=5e-5 \
--LEA 1 \
--lambda_flow_warp 0 \
--lambda_flow_warp_inv 2 \
--lambda_flow_warpx 0 \
--lambda_flow_warpx_inv 2 \
--debug 0 \
--load_gan_path 'stereogan_checkpoints/gan_driving_withflow_5.24_epoch30/ep30.pth' \
#2>&1 | tee ./logs/train-$model_name-$now.log &