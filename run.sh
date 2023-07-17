#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
model_name="VKITTI2_baseline_GMflow_lr_flow1e-4_bs4_from30_onlyinverse_flowinversewarp1"

python -u train.py \
--model_type='dispnet' \
--source_dataset='VKITTI2' \
--batch_size=4 \
--test_batch_size=4 \
--lr_rate=2e-4 \
--lr_gan=2e-4 \
--train_ratio_gan=3 \
--total_epochs=101 \
--save_interval=5 \
--print_freq=200 \
--checkpoint_save_path="stereogan_checkpoints/${model_name}" \
--load_checkpoints 1 \
--load_from_mgpus_model 1 \
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
--lr_flow=1e-4 \
--unimatch_stereo 0 \
--lambda_flow_warp 0 \
--lambda_flow_warp_inv 1 \
--lambda_flow_warpx 0 \
--lambda_flow_warpx_inv 1 \
--debug 0 \
--load_dispnet_path 'stereogan_checkpoints/VKITTI2_dispnet_pretrain/ep30_D1_0.0730_EPE1.3199_Thres2s0.1431_Thres4s0.0500_Thres5s0.0373.pth' \
--load_gan_path 'stereogan_checkpoints/gan_VKITTI2_withflow_fix/ep10.pth' \
--load_flownet_path 'stereogan_checkpoints/VKITTI2_GMflow_pretrain/ep29_epe_flow3.6163_f1_all0.1484_epe1_flow6.0657_fl_all10.2181.pth' \
2>&1 | tee ./logs/train-$model_name-$now.log &

