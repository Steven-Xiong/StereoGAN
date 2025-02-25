#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
model_name="flownets_baseline_VKITTI2"

python -u train_flownet_baseline.py \
--model_type='flownets' \
--source_dataset='VKITTI2' \
--batch_size=16 \
--test_batch_size=4 \
--total_epochs=101 \
--save_interval=5 \
--print_freq=220 \
--checkpoint_save_path="stereogan_checkpoints/${model_name}" \
--load_checkpoints 0 \
--load_from_mgpus_model 0 \
--writer=${model_name} \
--use_multi_gpu=0 \
--img_height=256 \
--img_width=512 \
--maxdisp=192 \
--flow=1 \
--lr_flow 2e-4 \
--unimatch_stereo 0 \
--debug 0 \
2>&1 | tee ./logs/train-$model_name-$now.log &
# --load_dispnet_path 'stereogan_checkpoints/StereoGAN_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_lr2e-5_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4/ep75_D1_0.3266_EPE3.4075_Thres2s0.4671_Thres4s0.2470_Thres5s0.1921_epe_flow7.3194_f1_all0.2723_epe1_flow12.6176_fl_all10.3904.pth' \
# --load_gan_path 'stereogan_checkpoints/StereoGAN_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_lr2e-5_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4/ep75_D1_0.3266_EPE3.4075_Thres2s0.4671_Thres4s0.2470_Thres5s0.1921_epe_flow7.3194_f1_all0.2723_epe1_flow12.6176_fl_all10.3904.pth' \
# --load_flownet_path 'stereogan_checkpoints/StereoGAN_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_lr2e-5_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4/ep75_D1_0.3266_EPE3.4075_Thres2s0.4671_Thres4s0.2470_Thres5s0.1921_epe_flow7.3194_f1_all0.2723_epe1_flow12.6176_fl_all10.3904.pth' \
#2>&1 | tee ./logs/train-$model_name-$now.log &
