import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
import os
import argparse
import sys
import itertools
import numpy as np
from scipy import misc
import numpy
import torch
import torch.nn as nn 
import torch.optim as  optim
import torch.nn.functional as F
from models.loss import warp_loss, model_loss0, PerceptualLoss, smooth_loss,flow_loss_func,flow_loss_func_val
from models.dispnet import dispnetcorr
from models.gan_nets import GeneratorResNet,GeneratorResNet_debug, Discriminator, weights_init_normal
from unimatch.unimatch import UniMatch
from dataset import ImageDataset, ValJointImageDataset, ImageDataset2, ValJointImageDataset2

from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils.util import AverageMeter
from utils.util import load_multi_gpu_checkpoint, load_checkpoint, load_multi_gpu_optimizer
from utils.metric_utils.metrics import *
from utils import pytorch_ssim
from consistency import apply_disparity,generate_image_left,generate_image_right
import cv2 as cv
#自己写的downsample
from models.bilinear_sampler import downsample_optical_flow
from utils.util import InputPadder
from utils.flow_viz import save_vis_flow_tofile
#valloader, net_flow, writer,  board_save=True
from IGEV.igev_stereo import IGEVStereo
from skimage import io
import skimage.io
import cv2

def val(args):
    writer = SummaryWriter(comment=args.writer)
    #os.makedirs(args.checkpoint_save_path, exist_ok=True)

    argsDict = args.__dict__
    for k,v in argsDict.items():
        writer.add_text('hyperparameter', '{} : {}'.format(str(k), str(v)))

    print_freq = args.print_freq
    test_freq = 1
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    # import IPython
    # IPython.embed()
    input_shape = (3, args.img_height, args.img_width)
    
    if args.IGEV:
        net = IGEVStereo(args)
    else:                 
        net = dispnetcorr(args.maxdisp)

    # add optical flow module
    if args.flow:
        net_flow = UniMatch(feature_channels=args.feature_channels,
                     num_scales=args.num_scales,
                     upsample_factor=args.upsample_factor,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_layers=args.num_transformer_layers,
                     reg_refine=args.reg_refine,
                     task='flow')

    
    if args.load_checkpoints:
        if args.load_from_mgpus_model:
            if args.load_IGEV_path:
                net = load_multi_gpu_checkpoint(net, args.load_IGEV_path, 'model')
            else:
                net.apply(weights_init_normal)
            # if args.load_flownet_path:
            #     net_flow = load_multi_gpu_checkpoint(net_flow,args.load_flownet_path,'model_flow')
            # else:
            #     net_flow.apply(weights_init_normal)
            
        else:
            if args.load_IGEV_path:
                net = load_checkpoint(net, args.load_checkpoint_path, device)
            else:
                net.apply(weights_init_normal)
            # if args.load_flownet_path:
            #     net_flow = load_checkpoint(net_flow, args.load_flownet_path, device)
            # else:
            #     net_flow.apply(weights_init_normal)   #可能有问题？
            

    # optimizer = optim.SGD(params, momentum=0.9)
    #加载optimizer的

    # if args.IGEV:
    #     optimizer = optim.AdamW(net.parameters(), lr=args.lr_rate,  #调一样不要整混
    #                             weight_decay=args.weight_decay_IGEV, eps=1e-8)
    # else:
    #     optimizer = optim.Adam(net.parameters(), lr=args.lr_rate, betas=(0.9, 0.999))
    # if args.flow:
    #     #optimizer_flow = optim.Adam(net_flow.parameters(), lr=args.lr_flow, betas=(0.9, 0.999))
    #     optimizer_flow = optim.AdamW(net_flow.parameters(),lr=args.lr_flow,weight_decay = args.weight_decay)
        
    # start epoch赋初值
    start_epoch = 0
    if args.load_checkpoints:
        print('load optimizer')
        checkpoint = torch.load(args.load_IGEV_path,map_location = device)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        start_step = checkpoint['step']
        
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.cuda()
        
        # if args.flow:
        #     optimizer_flow.load_state_dict(checkpoint['optimizer_flow_state_dict'])
        #     for state in optimizer_flow.state.values():
        #         for k, v in state.items():
        #             if torch.is_tensor(v):
        #                 state[k] = v.cuda()

    
    if args.use_multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=list(range(args.use_multi_gpu)))
        
    #     if args.flow:
    #         net_flow = nn.DataParallel(net_flow, device_ids=list(range(args.use_multi_gpu)))
    net.to(device)
    # if args.flow:
    #     net_flow.to(device)
        

    criterion_GAN = torch.nn.MSELoss().cuda()
    criterion_identity = torch.nn.L1Loss().cuda()
    ssim_loss = pytorch_ssim.SSIM()
    # add perceptual
    criterion_perceptual = PerceptualLoss().cuda()
    #criterion_smooth = smooth_loss().cuda()

    # data loader
    if args.source_dataset == 'driving':
        dataset = ImageDataset(height=args.img_height, width=args.img_width,left_right_consistency = args.left_right_consistency)
    elif args.source_dataset == 'VKITTI2':
        dataset = ImageDataset2(height=args.img_height, width=args.img_width,left_right_consistency = args.left_right_consistency)
    else:
        raise "No suportive dataset"
    #import pdb; pdb.set_trace()
    #dataset.get_item(1)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    valdataset = ValJointImageDataset()
    valdataset2 = ValJointImageDataset2()
    #valdataset2.getitem(1)
    valloader = torch.utils.data.DataLoader(valdataset, batch_size=args.test_batch_size, shuffle=False, num_workers=16)

    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    ## debug only
    #with torch.no_grad():
    #    l1_test_loss, out_val = val(valloader, net, G_AB, None, writer, epoch=0, board_save=True)
    #    val_loss_meter.update(l1_test_loss)
    #    print('Val epoch[{}/{}] loss: {}'.format(0, args.total_epochs, l1_test_loss))

    print('begin validation...')
    
    
    # TODO: 加 optical flow的evaluation metrics,看能不能相互促进
    with torch.no_grad():     
        net.eval()

        val_dataset = valdataset2
        print('Number of validation image pairs: %d' % len(val_dataset))

        out_list, epe_list = [], []
        results = {}
        print(len(val_dataset))
        #import pdb; pdb.set_trace()
        average_over_pixels= False
        padding_factor =32
        padding = True
        for val_id in range(len(val_dataset)):
            #import pdb;pdb.set_trace()
            left, right, disp_gt, left_forward,flow_gt,valid_gt = val_dataset[val_id]
  
            #visualization:
            # import cv2
            # image1_array = np.transpose(left, (1, 2, 0)).astype(np.uint8)
            # cv2.imwrite('image1.png', image1_array)
            flow_gt = flow_gt.transpose(2,0,1)

            left = torch.from_numpy(left).float()
            left_forward = torch.from_numpy(left_forward).float()
            right = torch.from_numpy(right).float()
            valid_gt = torch.from_numpy(valid_gt).float()
            flow_gt = torch.from_numpy(flow_gt).float()
            
            image1 = left[None].cuda()
            image2 = right[None].cuda()
            #print(image1.shape, image2.shape)
            #valid_gt = valid_gt.cuda()
            if padding:
                padder = InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
                image1, image2 = padder.pad(image1, image2)
            #import pdb; pdb.set_trace()
            if args.IGEV:
                disp_est = net(image1, image2, iters=args.valid_iters, test_mode=True)
            
            #flow_pr = results_dict['flow_preds'][-1]
                
            if padding:
                disp_est = padder.unpad(disp_est).cpu()
                
            else:
                disp_est = disp_est[0].cpu()
            #flow = padder.unpad(flow_pr[0]).cpu()
            
            if args.visualization:
                #import pdb; pdb.set_trace()
                frame_id = str(val_id) +'.png'
                #flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
                output_filename = os.path.join(args.output_path, frame_id)
                disp = disp_est.cpu().numpy().squeeze()
                #import pdb; pdb.set_trace()
                disp[disp<0]=0.0
                disp = np.round(disp * 256).astype(np.uint16)
                

                #disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

                disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
                disp_vis = disp_vis.astype("uint8")
                disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

                skimage.io.imsave(output_filename, disp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    # dataset
    parser.add_argument('--source_dataset', type=str, default='driving')
    parser.add_argument('--img_height', type=int, default=320)
    parser.add_argument('--img_width', type=int, default=512)

    # training
    parser.add_argument('--lr_rate', nargs='?', type=float, default=1e-4, help='learning rate for dispnetc')
    parser.add_argument('--lrepochs', type=str, default='30:1', help='the epochs to decay lr: the downscale rate')
    parser.add_argument('--lr_gan', nargs='?', type=float, default=2e-4, help='learning rate for GAN')
    parser.add_argument('--train_ratio_gan', nargs='?', type=int, default=5, help='training ratio disp:gan=5:1')
    parser.add_argument('--batch_size', nargs='?', type=int, default=6, help='batch size')
    parser.add_argument('--test_batch_size', nargs='?', type=int, default=4, help='test batch size')
    parser.add_argument('--total_epochs', nargs='?', type=int, default='201')
    parser.add_argument('--save_interval', nargs='?', type=int, default='10')
    parser.add_argument('--model_type', nargs='?', type=str, default='dispnetc')
    parser.add_argument('--maxdisp', type=int, default=192)

    # hyper params
    parser.add_argument('--lambda_cycle', type=float, default=10)
    parser.add_argument('--alpha_ssim', type=float, default=0.85)
    parser.add_argument('--lambda_id', type=float, default=5)
    parser.add_argument('--lambda_ms', type=float, default=1)
    parser.add_argument('--lambda_warp', type=float, default=0)
    parser.add_argument('--lambda_warp_inv', type=float, default=1)
    parser.add_argument('--lambda_disp_warp', type=float, default=0)
    parser.add_argument('--lambda_disp_warp_inv', type=float, default=1)
    parser.add_argument('--lambda_corr', type=float, default=10)

    parser.add_argument('--lambda_flow_warp', type=float,default = 0)
    parser.add_argument('--lambda_flow_warp_inv', type=float, default = 1)

    # load & save checkpoints
    parser.add_argument('--load_checkpoints', nargs='?', type=int, default=0, help='load from ckp(saved by Pytorch)')
    parser.add_argument('--load_from_mgpus_model', nargs='?', type=int, default=0, help='load ckp which is saved by mgus(nn.DataParallel)')
    parser.add_argument('--load_gan_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
    parser.add_argument('--load_dispnet_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
    parser.add_argument('--checkpoint_save_path', nargs='?', type=str, default='checkpoints/best_checkpoint.pth.tar')
    parser.add_argument('--load_flownet_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
    # tensorboard, print freq
    parser.add_argument('--writer', nargs='?', type=str, default='StereoGAN')
    parser.add_argument('--print_freq', '-p', default=150, type=int, metavar='N', help='print frequency (default: 150)')

    # other
    parser.add_argument('--use_multi_gpu', nargs='?', type=int, default=0, help='the number of multi gpu to use')

    # self add
    #parser.add_argument('--result_adv', type = float, default = 1)
    parser.add_argument('--cosine_similarity', type = float, default = 1)
    parser.add_argument('--perceptual', type = float, default =1 )
    parser.add_argument('--smooth_loss', type = float, default = 1)
    parser.add_argument('--left_right_consistency', type = float, default = 1)

    parser.add_argument('--flow', type = float,help= 'add optical flow branch', default = 1)
    parser.add_argument('--task', default='flow', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')
    parser.add_argument('--attn_type', default='swin', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')
    
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='exponential weighting')
    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--lr_flow', nargs='?', type=float, default=1e-4, help='learning rate for unimatch flow')
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # use unimatch stereo part
    parser.add_argument('--unimatch_stereo',type = float,help= 'use unimatch stereo as dispnet', default = 1)
    # parser.add_argument('--max_disp', default=400, type=int,
    #                     help='exclude very large disparity in the loss function')
    parser.add_argument('--debug', type=float, default=1)
    parser.add_argument('--visualization', type=float, default= 1)
    parser.add_argument('--output_path', type=str, default= 'visualize/stereo')

    parser.add_argument('--load_IGEV_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
    # use igev part
    parser.add_argument('--IGEV',type = float,help= 'use unimatch stereo as dispnet', default = 1)
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--val_iters', type=int, default=32, help="number of updates to the disparity field in each forward pass.")
    # IGEV Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')

    args = parser.parse_args()

    
    val(args)
