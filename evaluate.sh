#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
model_name="evaluate_gmflow"

python -u evaluate.py \
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
--load_checkpoints 1 \
--load_from_mgpus_model 2 \
--writer=${model_name} \
--use_multi_gpu=1 \
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
--unimatch_stereo 0 \
--lambda_flow_warp 0 \
--lambda_flow_warp_inv 0 \
--debug 0 \
--load_flownet_path '/home/x.zhexiao/StereoGAN/stereogan_checkpoints/ep70_D1_0.1089_EPE1.6608_Thres2s0.1937_Thres4s0.0747_Thres5s0.0559_epe_flow5.4063_f1_all0.2280_epe1_flow9.2990_fl_all10.3364.pth' 