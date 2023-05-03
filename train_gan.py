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
        checkpoint = torch.load(args.load_dispnet_path,map_location = device)

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
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    valdataset = ValJointImageDataset()
    valloader = torch.utils.data.DataLoader(valdataset, batch_size=args.test_batch_size, shuffle=False, num_workers=16)

    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    print('begin training...')
    print('start_epoch:', start_epoch)
    print('total_epoch:', args.total_epochs)
    best_val_d1 = 1.
    best_val_epe = 100.
    for epoch in range(start_epoch,args.total_epochs):
        #net.train()
        #G_AB.train()

        n_iter = 0
        running_loss = 0.
        t = time.time()
        # custom lr decay, or warm-up
        lr = args.lr_rate
        #TODO:改lr策略，用别的策略
        if args.unimatch_stereo:
            pass
        else:
            if epoch >= int(args.lrepochs.split(':')[0]):
                lr = lr / int(args.lrepochs.split(':')[1])
        #import pdb; pdb.set_trace()
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # add flow
        if args.flow:
            lr_flow = args.lr_flow
            if epoch >= int(args.lrepochs.split(':')[0]):
                lr_flow = lr_flow / int(args.lrepochs.split(':')[1])
            # for param_group in optimizer_flow.param_groups:
            #     param_group['lr'] = lr
        #import pdb; pdb.set_trace()
        for i, batch in enumerate(trainloader):
            n_iter += 1
            leftA = batch['leftA'].to(device)
            rightA = batch['rightA'].to(device)
            leftB = batch['leftB'].to(device)
            rightB = batch['rightB'].to(device)
            dispA = batch['dispA'].unsqueeze(1).float().to(device)
            dispB = batch['dispB'].to(device) 
            if args.left_right_consistency:
                error_mapB = batch['error_mapB'].to(device)
            leftA_forward = batch['leftA_forward'].to(device)
            leftB_forward = batch['leftB_forward'].to(device)
            flowA = batch['flowA'].to(device)
            flowB = batch['flowB'].to(device)
            if args.source_dataset == 'VKITTI2':
                validA = batch['validA'].to(device)
            validB = batch['validB'].to(device)
            out_shape = (leftA.size(0), 1, args.img_height//16, args.img_width//16)
            valid = torch.cuda.FloatTensor(np.ones(out_shape))
            fake = torch.cuda.FloatTensor(np.zeros(out_shape))
            
            #if i % args.train_ratio_gan == 0:
            # train generators
            G_AB.train()
            G_BA.train()
            
            optimizer_G.zero_grad()
            
            # Identity loss
            loss_id_A = (criterion_identity(G_BA(leftA), leftA) + criterion_identity(G_BA(rightA), rightA)) / 2
            loss_id_B = (criterion_identity(G_AB(leftB), leftB) + criterion_identity(G_AB(rightB), rightB)) / 2
            loss_id = (loss_id_A + loss_id_B) / 2

            if args.lambda_warp_inv:
                fake_leftB, fake_leftB_feats = G_AB(leftA, extract_feat=True)
                fake_leftA, fake_leftA_feats = G_BA(leftB, extract_feat=True)
                # add forward
                fake_leftA_forward, fake_leftA_forward_feats = G_BA(leftB_forward, extract_feat=True)
            else:
                fake_leftB = G_AB(leftA)
                fake_leftA = G_BA(leftB)
            if args.lambda_warp:
                fake_rightB, fake_rightB_feats = G_AB(rightA, extract_feat=True)
                fake_rightA, fake_rightA_feats = G_BA(rightB, extract_feat=True)
            else:
                fake_rightB = G_AB(rightA)
                fake_rightA = G_BA(rightB)
            loss_GAN_AB = criterion_GAN(D_B(fake_leftB), valid)
            loss_GAN_BA = criterion_GAN(D_A(fake_leftA), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            if args.lambda_warp_inv:
                rec_leftA, rec_leftA_feats = G_BA(fake_leftB, extract_feat=True)
            else:
                rec_leftA = G_BA(fake_leftB)
            if args.lambda_warp:
                rec_rightA, rec_rightA_feats = G_BA(fake_rightB, extract_feat=True)
            else:
                rec_rightA = G_BA(fake_rightB)
            rec_leftB = G_AB(fake_leftA)    #直接用
            rec_rightB = G_AB(fake_rightA)
            loss_cycle_A = (criterion_identity(rec_leftA, leftA) + criterion_identity(rec_rightA, rightA)) / 2
            loss_ssim_A = 1. - (ssim_loss(rec_leftA, leftA) + ssim_loss(rec_rightA, rightA)) / 2
            loss_cycle_B = (criterion_identity(rec_leftB, leftB) + criterion_identity(rec_rightB, rightB)) / 2
            loss_ssim_B = 1. - (ssim_loss(rec_leftB, leftB) + ssim_loss(rec_rightB, rightB)) / 2
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            loss_ssim = (loss_ssim_A + loss_ssim_B) / 2
            #print('loss_ssim', loss_ssim)
            # add cosine similarity loss
            #import pdb; pdb.set_trace()
            if args.cosine_similarity:
                #print('rec_leftA',rec_leftA.shape)
                #print('leftA',leftA.shape)
                loss_cosineA = 1- (F.cosine_similarity(rec_leftA, leftA,dim=-1).mean() + F.cosine_similarity(rec_leftA, leftA,dim=-1).mean()) / 2
                loss_cosineB = 1- (F.cosine_similarity(rec_leftB, leftB,dim=-1).mean() + F.cosine_similarity(rec_leftB, leftB,dim=-1).mean()) / 2
                loss_cosine = (loss_cosineA + loss_cosineB) /2
                #print(loss_cosine)
            else:
                loss_cosine = 0
            # add perceptual loss:
            if args.perceptual:
                loss_perceptualA = criterion_perceptual(rec_leftA, leftA).mean() 
                loss_perceptualB = criterion_perceptual(rec_leftB, leftB).mean() 
                loss_perceptual = (loss_perceptualA + loss_perceptualB)/2
            else:
                loss_perceptual = 0 

            # mode seeking loss
            if args.lambda_ms:
                loss_ms = G_AB(leftA, zx=True, zx_relax=True).mean()
            else:
                loss_ms = 0

            # warping loss
            if args.lambda_warp_inv:
                fake_leftB_warp, loss_warp_inv_feat1 = G_AB(rightA, -dispA, True, [x.detach() for x in fake_leftB_feats])
                rec_leftA_warp, loss_warp_inv_feat2 = G_BA(fake_rightB, -dispA, True, [x.detach() for x in rec_leftA_feats])
                loss_warp_inv1 = warp_loss([(G_BA(fake_leftB_warp[0]), fake_leftB_warp[1])], [leftA], weights=[1])
                loss_warp_inv2 = warp_loss([rec_leftA_warp], [leftA], weights=[1])
                #print(len(rec_leftA_warp),len(leftA))
                #print(rec_leftA_warp[0].shape,rec_leftA_warp[1].shape,leftA.shape)
                loss_warp_inv = loss_warp_inv1 + loss_warp_inv2 + loss_warp_inv_feat1.mean() + loss_warp_inv_feat2.mean()
            else:
                loss_warp_inv = 0

            if args.lambda_warp:
                fake_rightB_warp, loss_warp_feat1 = G_AB(leftA, dispA, True, [x.detach() for x in fake_rightB_feats])
                rec_rightA_warp, loss_warp_feat2 = G_BA(fake_leftB, dispA, True, [x.detach() for x in rec_rightA_feats])
                loss_warp1 = warp_loss([(G_BA(fake_rightB_warp[0]), fake_rightB_warp[1])], [rightA], weights=[1])
                loss_warp2 = warp_loss([rec_rightA_warp], [rightA], weights=[1])
                loss_warp = loss_warp1 + loss_warp2 + loss_warp_feat1.mean() + loss_warp_feat2.mean()
            else:
                loss_warp = 0

            lambda_ms = args.lambda_ms * (args.total_epochs - epoch) / args.total_epochs
            loss_G = loss_GAN + args.lambda_cycle*(args.alpha_ssim*loss_ssim+(1-args.alpha_ssim)*loss_cycle) + args.lambda_id*loss_id \
                    + args.lambda_warp*loss_warp + args.lambda_warp_inv*loss_warp_inv  + lambda_ms*loss_ms \
                    + args.cosine_similarity * loss_cosine + args.perceptual*loss_perceptual #+ args.smooth_loss * loss_smooth
            loss_G.backward()
            optimizer_G.step()
            
            # train discriminators. A: real, B: syn
            optimizer_D_A.zero_grad()
            loss_real_A = criterion_GAN(D_A(leftA), valid)
            fake_leftA.detach_()
            loss_fake_A = criterion_GAN(D_A(fake_leftA), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            
            optimizer_D_B.zero_grad()
            #loss_real_B = criterion_GAN(D_B(torch.cat([syn_left_img, syn_right_img], 0)), valid)
            #fake_syn_left.detach_()
            #fake_syn_right.detach_()
            #loss_fake_B = criterion_GAN(D_B(torch.cat([fake_syn_left, fake_syn_right], 0)), fake)
            loss_real_B = criterion_GAN(D_B(leftB), valid)
            fake_leftB.detach_()
            loss_fake_B = criterion_GAN(D_B(fake_leftB), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
                
                # add flow forward and backward generalization:
                
            

            if i % print_freq == print_freq - 1:
                print('epoch[{}/{}]  step[{}/{}]  loss_G: {}'.format(epoch, args.total_epochs, i, len(trainloader), loss_G.item() ))
                train_loss_meter.update(running_loss / print_freq)

                writer.add_scalar('loss/loss_G', loss_G, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_gan', loss_GAN, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_cycle', loss_cycle, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_id', loss_id, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_warp', loss_warp, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_warp_inv', loss_warp_inv, train_loss_meter.count * print_freq)
                #writer.add_scalar('loss/loss_corr', loss_corr, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_ms', loss_ms, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_D_A', loss_D_A, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_D_B', loss_D_B, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_cosine', loss_cosine, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_perceptual', loss_perceptual, train_loss_meter.count * print_freq)
                # writer.add_scalar('loss/loss_smooth', loss_smooth, train_loss_meter.count * print_freq)
                

                imgA_visual = vutils.make_grid(leftA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                fakeB_visual = vutils.make_grid(fake_leftB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                recA_visual = vutils.make_grid(rec_leftA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                rightA_visual = vutils.make_grid(rightA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                fakeB_R_visual = vutils.make_grid(fake_rightB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                recA_R_visual = vutils.make_grid(rec_rightA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)

                imgB_visual = vutils.make_grid(leftB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                fakeA_visual = vutils.make_grid(fake_leftA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                recB_visual = vutils.make_grid(rec_leftB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                rightB_visual = vutils.make_grid(rightB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                fakeA_R_visual = vutils.make_grid(fake_rightA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                recB_R_visual = vutils.make_grid(rec_rightB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)

                writer.add_image('ABA_L/imgA', imgA_visual, i)
                writer.add_image('ABA_L/fakeB', fakeB_visual, i)
                writer.add_image('ABA_L/recA', recA_visual, i)
                writer.add_image('ABA_R/imgA', rightA_visual, i)
                writer.add_image('ABA_R/fakeB', fakeB_R_visual, i)
                writer.add_image('ABA_R/recA', recA_R_visual, i)
                writer.add_image('BAB_L/imgB', imgB_visual, i)
                writer.add_image('BAB_L/fakeA', fakeA_visual, i)
                writer.add_image('BAB_L/recB', recB_visual, i)
                writer.add_image('BAB_R/imgB', rightB_visual, i)
                writer.add_image('BAB_R/fakeA', fakeA_R_visual, i)
                writer.add_image('BAB_R/recB', recB_R_visual, i)

        t1 = time.time()   #to do: add other evaluation metrics
        print('epoch:{}, cost time:{} '.format(epoch, t1-t))
        # add flow
        if (epoch % args.save_interval == 0):
    
            torch.save({
                    'epoch': epoch,
                    'G_AB': G_AB.state_dict(),
                    'G_BA': G_BA.state_dict(),
                    'D_A': D_A.state_dict(),
                    'D_B': D_B.state_dict(),
                    'optimizer_DA_state_dict': optimizer_D_A.state_dict(),
                    'optimizer_DB_state_dict': optimizer_D_B.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    }, args.checkpoint_save_path + '/ep' + str(epoch) + '.pth')

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
    args = parser.parse_args()

    torch.manual_seed(3407)
    np.random.seed(3407)
    
    train(args)
