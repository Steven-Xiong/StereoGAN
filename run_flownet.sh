#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
model_name="diving_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_lr2e-5_gan2e-5_baseline_flownets_lr_flow1e-4_bs4"

python -u train_flownet.py \
--model_type='dispnet' \
--source_dataset='driving' \
--batch_size=4 \
--test_batch_size=4 \
--lr_rate=2e-5 \
--lr_gan=2e-5 \
--train_ratio_gan=3 \
--total_epochs=101 \
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
--lr_flow 1e-4 \
--unimatch_stereo 0 \
--lambda_flow_warp 1 \
--lambda_flow_warp_inv 1 \
--debug 0 \
# --load_dispnet_path 'stereogan_checkpoints/StereoGAN_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_lr2e-5_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4/ep75_D1_0.3266_EPE3.4075_Thres2s0.4671_Thres4s0.2470_Thres5s0.1921_epe_flow7.3194_f1_all0.2723_epe1_flow12.6176_fl_all10.3904.pth' \
# --load_gan_path 'stereogan_checkpoints/StereoGAN_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_lr2e-5_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4/ep75_D1_0.3266_EPE3.4075_Thres2s0.4671_Thres4s0.2470_Thres5s0.1921_epe_flow7.3194_f1_all0.2723_epe1_flow12.6176_fl_all10.3904.pth' \
# --load_flownet_path 'stereogan_checkpoints/StereoGAN_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_lr2e-5_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4/ep75_D1_0.3266_EPE3.4075_Thres2s0.4671_Thres4s0.2470_Thres5s0.1921_epe_flow7.3194_f1_all0.2723_epe1_flow12.6176_fl_all10.3904.pth' \
#2>&1 | tee ./logs/train-$model_name-$now.log &
