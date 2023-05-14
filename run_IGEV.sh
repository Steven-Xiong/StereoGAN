#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
model_name="driving_IGEV_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_IGEVlr1e-4_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4_warp5_IGEV_withpretrain50_try"

python -u train_IGEV.py \
--model_type='IGEV' \
--source_dataset='driving' \
--batch_size=2 \
--test_batch_size=2 \
--lr_rate=1e-4 \
--lr_gan=2e-5 \
--train_ratio_gan=3 \
--total_epochs=101 \
--save_interval=5 \
--print_freq=20 \
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
--lr_flow=1e-4 \
--IGEV 1 \
--lambda_flow_warp 5 \
--lambda_flow_warp_inv 5 \
--lambda_flow_warpx 5 \
--lambda_flow_warpx_inv 5 \
--debug 0 \
#--load_IGEV_path 'stereogan_checkpoints/ep50_D1_0.1843_EPE2.4390_Thres2s0.3066_Thres4s0.1276_Thres5s0.0953.pth' \
#--load_gan_path 'stereogan_checkpoints/gan_dispnet_driving/ep10.pth' \
#--load_flownet_path 'stereogan_checkpoints/ep60_epe_flow8.5246_f1_all0.3613_epe1_flow14.3549_fl_all10.5211.pth' \
# 2>&1 | tee ./logs/train-$model_name-$now.log &
#--load_flownet_path 'stereogan_checkpoints/ep60_epe_flow8.5246_f1_all0.3613_epe1_flow14.3549_fl_all10.5211.pth' \
# --load_IGEV_path 'stereogan_checkpoints/driving_dispnet_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_IGEVlr1e-4_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4_warp5_IGEV_withpretrain50/ep54_D1_0.1697_EPE2.2416_Thres2s0.2861_Thres4s0.1165_Thres5s0.0847_epe_flow8.2044_f1_all0.3129_epe1_flow13.8658_fl_all10.4400.pth' \
# --load_gan_path 'stereogan_checkpoints/driving_dispnet_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_IGEVlr1e-4_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4_warp5_IGEV_withpretrain50/ep54_D1_0.1697_EPE2.2416_Thres2s0.2861_Thres4s0.1165_Thres5s0.0847_epe_flow8.2044_f1_all0.3129_epe1_flow13.8658_fl_all10.4400.pth' \
# --load_flownet_path 'stereogan_checkpoints/driving_dispnet_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_IGEVlr1e-4_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4_warp5_IGEV_withpretrain50/ep54_D1_0.1697_EPE2.2416_Thres2s0.2861_Thres4s0.1165_Thres5s0.0847_epe_flow8.2044_f1_all0.3129_epe1_flow13.8658_fl_all10.4400.pth' \
# 2>&1 | tee ./logs/train-$model_name-$now.log &
#--load_IGEV_path 'stereogan_checkpoints/driving_dispnet_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_IGEVlr1e-4_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4_warp5_IGEV/ep12_D1_0.2277_EPE2.6143_Thres2s0.3806_Thres4s0.1471_Thres5s0.1000_epe_flow10.8877_f1_all0.4376_epe1_flow17.8709_fl_all10.5885.pth' \
#--load_gan_path 'stereogan_checkpoints/driving_dispnet_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_IGEVlr1e-4_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4_warp5_IGEV/ep12_D1_0.2277_EPE2.6143_Thres2s0.3806_Thres4s0.1471_Thres5s0.1000_epe_flow10.8877_f1_all0.4376_epe1_flow17.8709_fl_all10.5885.pth' \
#--load_flownet_path 'stereogan_checkpoints/driving_dispnet_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_warp5_dispwarp5_imwidth512_height256_ep100_IGEVlr1e-4_gan2e-5_baseline_GMflow_lr_flow1e-4_bs4_warp5_IGEV/ep12_D1_0.2277_EPE2.6143_Thres2s0.3806_Thres4s0.1471_Thres5s0.1000_epe_flow10.8877_f1_all0.4376_epe1_flow17.8709_fl_all10.5885.pth' \
#dataset: 'driving', 'VKITTI2'