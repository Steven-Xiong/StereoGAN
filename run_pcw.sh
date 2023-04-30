#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_lr2e-5_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4_warp5_PCW"

python -u train_pcwnet.py \
--model_type='pcwnet' \
--source_dataset='driving' \
--batch_size=2 \
--test_batch_size=2 \
--lr_rate=2e-5 \
--lr_gan=2e-5 \
--train_ratio_gan=3 \
--total_epochs=111 \
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
--lambda_corr=1 \
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
--left_right_consistency=0 \
--flow=1 \
--num_scales 2 \
--upsample_factor 4 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--lr_flow 1e-4 \
--pcwnet 1 \
--lambda_flow_warp 5 \
--lambda_flow_warp_inv 5 \
--debug 1 \
#2>&1 | tee ./logs/train-$model_name-$now.log &

