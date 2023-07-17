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
from dataset import ImageDataset, ValJointImageDataset, ImageDataset2

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

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips
import cv2 as cv

def val(valloader, G_AB,G_BA, D_A, D_B, writer, epoch=1, board_save=True):
    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()
    
    PSNR, SSIM, LPIPS = 0,0,0
    i = 0
    loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization
    for left_img, right_img, disp_gt, left_forward, flow, valid in valloader:
        left_img = left_img.cuda()
        right_img = right_img.cuda()
        disp_gt = disp_gt.cuda()
        i = i + 1
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        
        #fake_leftB = G_AB(leftA)
        fake_leftA = G_BA(left_img)
        rec_leftB = G_AB(fake_leftA) 
        #import pdb; pdb.set_trace()
        # left_img= torch.squeeze(left_img, 0).cpu().numpy()
        # rec_leftB = torch.squeeze(rec_leftB,0).cpu().numpy()
        
        ssim_batch = 0
        psnr_batch = 0
        lpips_batch = 0
        left_img1= left_img.cpu().numpy().transpose(0,2,3,1)
        rec_leftB1 = rec_leftB.cpu().numpy().transpose(0,2,3,1)
        
        for k in range(left_img.shape[0]):
            #import pdb; pdb.set_trace()

            psnr_batch += compute_psnr((left_img1[k,:,:,:]+1.0)/2.0, (rec_leftB1[k,:,:,:]+1.0)/2.0, data_range = 1.0)
            ssim_batch += compute_ssim((left_img1[k,:,:,:]+1.0)/2.0, (rec_leftB1[k,:,:,:]+1.0)/2.0, multichannel=True)
            
            #cv.imwrite('left.png',(left_img1[k,:,:,:]+1.0)*255/2.0)
            #cv.imwrite('rec_leftB1.png',(rec_leftB1[k,:,:,:]+1.0)*255/2.0)
            #lpips_batch += loss_fn_vgg(left_img[k,:,:,:], rec_leftB[k,:,:,:])
            
            
        ssim_batch /= float(left_img.shape[0])
        psnr_batch /= float(left_img.shape[0])
        lpips_batch = loss_fn_vgg(left_img, rec_leftB)
        
        print(psnr_batch, ssim_batch,lpips_batch.mean())
        #import pdb; pdb.set_trace()
        PSNR += psnr_batch
        SSIM += ssim_batch
        LPIPS += lpips_batch.mean()
               
    if board_save:
        writer.add_scalar("val/PSNR", PSNR/i, epoch)
        writer.add_scalar("val/SSIM", SSIM/i, epoch)
        writer.add_scalar("val/LPIPS", LPIPS/i, epoch)
        
    return PSNR/len(valloader), SSIM/len(valloader), LPIPS/len(valloader)


# def test(trainloader, G_AB,G_BA, D_A, D_B, writer, epoch=1, board_save=True):
#     ssim_batch = 0
#     psnr_batch = 0
#     lpips_batch = 0
#     fake_leftB1= img1.cpu().numpy().transpose(0,2,3,1)
#     leftB1 = img2.cpu().numpy().transpose(0,2,3,1)        
#     for k in range(img1.shape[0]):
#         #import pdb; pdb.set_trace()

#         psnr_batch += compute_psnr((fake_leftB1[k,:,:,:]+1.0)/2.0, (leftB1[k,:,:,:]+1.0)/2.0, data_range = 1.0)
#         ssim_batch += compute_ssim((fake_leftB1[k,:,:,:]+1.0)/2.0, (leftB1[k,:,:,:]+1.0)/2.0, multichannel=True)
        
#         cv.imwrite('left.png',(leftB1[k,:,:,:]+1.0)*255/2.0)
#         cv.imwrite('rec_leftB1.png',(leftB1[k,:,:,:]+1.0)*255/2.0)
#         #lpips_batch += loss_fn_vgg(left_img[k,:,:,:], rec_leftB[k,:,:,:])
#     ssim_batch /= float(img1.shape[0])
#     psnr_batch /= float(img1.shape[0])
#     lpips_batch = loss_fn_vgg(img1, img2)
    
#     print(psnr_batch, ssim_batch,lpips_batch.mean())
#     #import pdb; pdb.set_trace()
#     PSNR += psnr_batch
#     SSIM += ssim_batch
#     LPIPS += lpips_batch.mean()
    
#     return PSNR/len(trainloader), SSIM/len(trainloader), LPIPS/len(trainloader)

def val_AB(trainloader, G_AB,G_BA, D_A, D_B, writer, epoch=1, board_save=True):
    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()
    
    PSNR_AB, SSIM_AB, LPIPS_AB,PSNR_BA, SSIM_BA, LPIPS_BA = 0,0,0,0,0,0
    i = 0
    loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization
    for i, batch in enumerate(trainloader):
        leftA = batch['leftA'].to(device)
        rightA = batch['rightA'].to(device)
        leftB = batch['leftB'].to(device)
        rightB = batch['rightB'].to(device)

        #fake_leftB = G_AB(leftA)
        fake_leftA = G_BA(leftB)
        fake_leftB = G_AB(leftA) 
        #import pdb; pdb.set_trace()
        # left_img= torch.squeeze(left_img, 0).cpu().numpy()
        # rec_leftB = torch.squeeze(rec_leftB,0).cpu().numpy()
        
        ssimAB_batch = 0
        psnrAB_batch = 0
        lpipsAB_batch = 0
        ssimBA_batch = 0
        psnrBA_batch = 0
        lpipsBA_batch = 0
        
        leftA1= leftA.cpu().numpy().transpose(0,2,3,1)
        leftB1 = leftB.cpu().numpy().transpose(0,2,3,1)
        fake_leftA1 = fake_leftA.cpu().numpy().transpose(0,2,3,1)
        fake_leftB1 = fake_leftB.cpu().numpy().transpose(0,2,3,1)
        
        for k in range(leftA.shape[0]):
            #import pdb; pdb.set_trace()
            psnrBA_batch += compute_psnr((leftB1[k,:,:,:]+1.0)/2.0, (fake_leftA1[k,:,:,:]+1.0)/2.0, data_range = 1.0)
            ssimBA_batch += compute_ssim((leftB1[k,:,:,:]+1.0)/2.0, (fake_leftA1[k,:,:,:]+1.0)/2.0, multichannel=True)
            psnrAB_batch += compute_psnr((leftA1[k,:,:,:]+1.0)/2.0, (fake_leftB1[k,:,:,:]+1.0)/2.0, data_range = 1.0)
            ssimAB_batch += compute_ssim((leftA1[k,:,:,:]+1.0)/2.0, (fake_leftB1[k,:,:,:]+1.0)/2.0, multichannel=True)
            
            #cv.imwrite('leftA.png',(leftA1[k,:,:,:]+1.0)*255/2.0)
            #cv.imwrite('fake_leftA1.png',(fake_leftA1[k,:,:,:]+1.0)*255/2.0)
            #cv.imwrite('leftB.png',(leftB1[k,:,:,:]+1.0)*255/2.0)
            #cv.imwrite('fake_leftB1.png',(fake_leftB1[k,:,:,:]+1.0)*255/2.0)
            
            
        psnrAB_batch /= float(leftA.shape[0])
        psnrBA_batch /= float(leftA.shape[0])    
        ssimAB_batch /= float(leftA.shape[0])
        ssimBA_batch /= float(leftA.shape[0])
        
        lpipsAB_batch = loss_fn_vgg(leftB, fake_leftA)
        lpipsBA_batch = loss_fn_vgg(leftA, fake_leftB)
        
        #print(psnrAB_batch, ssimAB_batch,lpipsAB_batch.mean())
        #import pdb; pdb.set_trace()
        PSNR_AB += psnrAB_batch
        SSIM_AB += ssimAB_batch
        LPIPS_AB += lpipsAB_batch.mean()
        PSNR_BA += psnrBA_batch
        SSIM_BA += ssimBA_batch
        LPIPS_BA += lpipsBA_batch.mean()
               
    # if board_save:
    #     writer.add_scalar("val/PSNR", PSNR/i, epoch)
    #     writer.add_scalar("val/SSIM", SSIM/i, epoch)
    #     writer.add_scalar("val/LPIPS", LPIPS/i, epoch)
        
    return PSNR_AB/len(trainloader), SSIM_AB/len(trainloader), LPIPS_AB/len(trainloader),PSNR_BA/len(trainloader), SSIM_BA/len(trainloader), LPIPS_BA/len(trainloader)


def train(args):
    writer = SummaryWriter(comment=args.writer)
    os.makedirs(args.checkpoint_save_path, exist_ok=True)

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

    G_AB = GeneratorResNet(input_shape, 2)    # 定义用到的generative model
    G_BA = GeneratorResNet(input_shape, 2)
    D_A = Discriminator(3)
    D_B = Discriminator(3)
    if args.debug:
        G_AB_debug = GeneratorResNet_debug(input_shape, 2)
        G_BA_debug = GeneratorResNet_debug(input_shape, 2)
    # if args.flow:
    #     G_A_forward = GeneratorResNet(input_shape,2)   #加前向后向生成
    #     G_A_backward = GeneratorResNet(input_shape,2)
    #     D_A_forward = Discriminator(3)
    #     D_A_backward = Discriminator(3)
    if args.load_checkpoints:
        if args.load_from_mgpus_model:
          
            G_AB = load_multi_gpu_checkpoint(G_AB, args.load_gan_path, 'G_AB')
            G_BA = load_multi_gpu_checkpoint(G_BA, args.load_gan_path, 'G_BA')
            D_A = load_multi_gpu_checkpoint(D_A, args.load_gan_path, 'D_A')
            D_B = load_multi_gpu_checkpoint(D_B, args.load_gan_path, 'D_B')
        else:
    
            G_AB = load_checkpoint(G_AB, args.load_gan_path, 'G_AB')
            G_BA = load_checkpoint(G_BA, args.load_gan_path, 'G_BA')
            D_A = load_checkpoint(D_A, args.load_gan_path, 'D_A')
            D_B = load_checkpoint(D_B, args.load_gan_path, 'D_B')
    else:
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)
        if args.debug:
            G_AB_debug.apply(weights_init_normal)
            G_BA_debug.apply(weights_init_normal)

    # optimizer = optim.SGD(params, momentum=0.9)
    #加载optimizer的
    
    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr_gan, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=args.lr_gan, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=args.lr_gan, betas=(0.5, 0.999))
    
    # start epoch赋初值
    start_epoch = 0
    if args.load_checkpoints:
        print('load optimizer')
        checkpoint = torch.load(args.load_gan_path,map_location = device)

        start_epoch = checkpoint['epoch']+1
        
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        for state in optimizer_G.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        optimizer_D_A.load_state_dict(checkpoint['optimizer_DA_state_dict'])
        for state in optimizer_D_A.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        optimizer_D_B.load_state_dict(checkpoint['optimizer_DB_state_dict'])
        for state in optimizer_D_B.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        # optimizer = load_multi_gpu_optimizer(args.load_dispnet_path,'optimizer_state_dict')
        # optimizer_G = load_multi_gpu_optimizer(args.load_gan_path,'optimizer_G_state_dict')
        # optimizer_D_A = load_multi_gpu_optimizer(args.load_gan_path,'optimizer_DA_state_dict')
        # optimizer_D_B = load_multi_gpu_optimizer(args.load_gan_path,'optimizer_DB_state_dict')
        # if args.flow:
        #     optimizer_flow = load_multi_gpu_optimizer(args.load_flownet_path,'optimizer_flow_state_dict')
        #import pdb; pdb.set_trace()
    
    if args.use_multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        G_AB = nn.DataParallel(G_AB, device_ids=list(range(args.use_multi_gpu)))
        G_BA = nn.DataParallel(G_BA, device_ids=list(range(args.use_multi_gpu)))
        D_A = nn.DataParallel(D_A, device_ids=list(range(args.use_multi_gpu)))
        D_B = nn.DataParallel(D_B, device_ids=list(range(args.use_multi_gpu)))
        if args.debug:
            G_AB_debug = nn.DataParallel(G_AB_debug,device_ids=list(range(args.use_multi_gpu)))
            G_BA_debug = nn.DataParallel(G_BA_debug,device_ids=list(range(args.use_multi_gpu)))
        
    #net.to(device)
    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)
    if args.debug:
        G_AB_debug.to(device)
        G_BA_debug.to(device)


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
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    valdataset = ValJointImageDataset()
    valloader = torch.utils.data.DataLoader(valdataset, batch_size=args.test_batch_size, shuffle=False, num_workers=16)

    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    print('begin training...')
    print('start_epoch:', start_epoch)
    print('total_epoch:', args.total_epochs)
    best_val_d1 = 1.
    best_val_epe = 100.
    
    # 对training set进行test
        
    #import pdb; pdb.set_trace()
    PSNR = 0
    SSIM = 0
    LPIPS = 0  
    #loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization
    
        # test code
        # TODO: 加 optical flow的evaluation metrics,看能不能相互促进
        
    with torch.no_grad():
        print(len(trainloader))
        PSNR_AB, SSIM_AB, LPIPS_AB,PSNR_BA, SSIM_BA, LPIPS_BA=val_AB(trainloader, G_AB,G_BA, D_A, D_B, writer, epoch=1, board_save=True)    
        print('test psnr A2B: ', PSNR_AB, 'test ssim A2B: ',SSIM_AB, ' test lpips A2B: ',LPIPS_AB)
        print('test psnr B2A: ', PSNR_BA, 'test ssim B2A: ',SSIM_BA, ' test lpips B2A: ',LPIPS_BA)
    with torch.no_grad():
        print(len(valloader))
        psnr_score, ssim_score, lpips_score = val(valloader, G_AB,G_BA, D_A, D_B, writer, epoch=1, board_save=True)
        print('test psnr: ', psnr_score, 'test ssim: ',ssim_score, ' test lpips: ',lpips_score)
            
                
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
    parser.add_argument('--test_batch_size', nargs='?', type=int, default=8, help='test batch size')
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
    # add flow warp x:
    parser.add_argument('--lambda_flow_warpx', type = float, default= 0)
    parser.add_argument('--lambda_flow_warpx_inv',type = float, default=0)

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
    parser.add_argument('--left_right_consistency', type = float, default = 0)

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
    args = parser.parse_args()

    torch.manual_seed(3407)
    np.random.seed(3407)
    
    train(args)
