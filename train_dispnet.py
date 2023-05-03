import time
import os
import argparse
import sys
import itertools
import numpy as np
from scipy import misc

import torch
import torch.nn as nn 
import torch.optim as  optim
import torch.nn.functional as F
from models.loss import warp_loss, model_loss0
from models.dispnet import dispnetcorr
from models.gan_nets import GeneratorResNet, Discriminator, weights_init_normal
from dataset import ImageDataset, ValJointImageDataset, ImageDataset2

from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils.util import AverageMeter
from utils.util import load_multi_gpu_checkpoint, load_checkpoint
from utils.metric_utils.metrics import *
from utils import pytorch_ssim

def val(valloader, net, writer, epoch=1, board_save=True):
    net.eval()
    EPEs, D1s, Thres1s, Thres2s, Thres3s = 0, 0, 0, 0, 0
    i = 0
    for left_img, right_img, disp_gt, left_forward, flow, valid in valloader:
        left_img = left_img.cuda()
        right_img = right_img.cuda()
        disp_gt = disp_gt.cuda()
        i = i + 1
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        disp_est = net(left_img, right_img)[0].squeeze(1)
        EPEs += EPE_metric(disp_est, disp_gt, mask)
        D1s += D1_metric(disp_est, disp_gt, mask)
        Thres1s += Thres_metric(disp_est, disp_gt, mask, 2.0)
        Thres2s += Thres_metric(disp_est, disp_gt, mask, 4.0)
        Thres3s += Thres_metric(disp_est, disp_gt, mask, 5.0)
    if board_save:
        writer.add_scalar("val/EPE", EPEs/i, epoch)
        writer.add_scalar("val/D1", D1s/i, epoch)
        writer.add_scalar("val/Thres2", Thres1s/i, epoch)
        writer.add_scalar("val/Thres4", Thres2s/i, epoch)
        writer.add_scalar("val/Thres5", Thres3s/i, epoch)
    return EPEs/i, D1s/i, Thres1s/i,Thres2s/i,Thres3s/i

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
    
    input_shape = (3, args.img_height, args.img_width)
    net = dispnetcorr(args.maxdisp)

    if args.load_checkpoints:
        if args.load_from_mgpus_model:
            if args.load_dispnet_path:
                net = load_multi_gpu_checkpoint(net, args.load_dispnet_path, 'model')
            else:
                net.apply(weights_init_normal)
            
        else:
            if args.load_dispnet_path:
                net = load_checkpoint(net, args.load_checkpoint_path, device)
            else:
                net.apply(weights_init_normal)
    else:
        net.apply(weights_init_normal)

    # optimizer = optim.SGD(params, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=args.lr_rate, betas=(0.9, 0.999))
    
    # start epoch赋初值
    start_epoch = 0
    if args.load_checkpoints:
        print('load optimizer')
        checkpoint = torch.load(args.load_dispnet_path,map_location = device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    if args.use_multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=list(range(args.use_multi_gpu)))
        

    net.to(device)

    # data loader
    if args.source_dataset == 'driving':
        dataset = ImageDataset(height=args.img_height, width=args.img_width,left_right_consistency = args.left_right_consistency)
    elif args.source_dataset == 'VKITTI2':
        dataset = ImageDataset2(height=args.img_height, width=args.img_width,left_right_consistency = args.left_right_consistency)
    else:
        raise "No suportive dataset"
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valdataset = ValJointImageDataset()
    valloader = torch.utils.data.DataLoader(valdataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)

    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    ## debug only
    #with torch.no_grad():
    #    l1_test_loss, out_val = val(valloader, net, G_AB, None, writer, epoch=0, board_save=True)
    #    val_loss_meter.update(l1_test_loss)
    #    print('Val epoch[{}/{}] loss: {}'.format(0, args.total_epochs, l1_test_loss))

    print('begin training...')
    best_val_d1 = 1.
    best_val_epe = 100.
    for epoch in range(args.total_epochs):

        n_iter = 0
        running_loss = 0.
        t = time.time()
        # custom lr decay, or warm-up
        lr = args.lr_rate
        # if epoch >= int(args.lrepochs.split(':')[0]):
        #     lr = lr / int(args.lrepochs.split(':')[1])
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        for i, batch in enumerate(trainloader):
            n_iter += 1
            leftA = batch['leftA'].to(device)
            rightA = batch['rightA'].to(device)
            leftB = batch['leftB'].to(device)
            rightB = batch['rightB'].to(device)
            dispA = batch['dispA'].unsqueeze(1).float().to(device)
            dispB = batch['dispB'].to(device) 
            out_shape = (leftA.size(0), 1, args.img_height//16, args.img_width//16)
            valid = torch.cuda.FloatTensor(np.ones(out_shape))
            fake = torch.cuda.FloatTensor(np.zeros(out_shape))
            
            # train disp net
            net.train()
        
            optimizer.zero_grad()
            disp_ests = net(leftA, rightA)
            mask = (dispA < args.maxdisp) & (dispA > 0)
            loss0 = model_loss0(disp_ests, dispA, mask)

            loss = loss0 
            loss.backward()
            optimizer.step()

            if i % print_freq == print_freq - 1:
                print('epoch[{}/{}]  step[{}/{}]  loss: {}'.format(epoch, args.total_epochs, i, len(trainloader), loss.item() ))
                train_loss_meter.update(running_loss / print_freq)
                #writer.add_scalar('loss/trainloss avg_meter', train_loss_meter.val, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_disp', loss0, train_loss_meter.count * print_freq)


        with torch.no_grad():
            EPE, D1,Thres1s,Thres2s,Thres3s = val(valloader, net, writer, epoch=epoch, board_save=True)
        t1 = time.time()
        print('epoch:{}, D1:{:.4f}, EPE:{:.4f},Thres2s:{:.4f},Thres4s:{:.4f},Thres5s:{:.4f}, cost time:{} '.format(epoch, D1, EPE,Thres1s,Thres2s,Thres3s, t1-t))

        if (epoch % args.save_interval == 0) or D1 < best_val_d1 or EPE < best_val_epe:
            best_val_d1 = D1
            best_val_epe = EPE
            torch.save({
                        'epoch': epoch,
                        'model': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, args.checkpoint_save_path + '/ep' + str(epoch) + '_D1_{:.4f}_EPE{:.4f}_Thres2s{:.4f}_Thres4s{:.4f}_Thres5s{:.4f}'.format(D1, EPE,Thres1s,Thres2s,Thres3s) + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    # dataset
    parser.add_argument('--source_dataset', type=str, default='driving')
    parser.add_argument('--img_height', type=int, default=320)
    parser.add_argument('--img_width', type=int, default=512)

    # training
    parser.add_argument('--lr_rate', nargs='?', type=float, default=1e-4, help='learning rate for dispnetc')
    parser.add_argument('--lrepochs', type=str, default='30:1', help='the epochs to decay lr: the downscale rate')
    parser.add_argument('--batch_size', nargs='?', type=int, default=6, help='batch size')
    parser.add_argument('--test_batch_size', nargs='?', type=int, default=4, help='test batch size')
    parser.add_argument('--total_epochs', nargs='?', type=int, default='201')
    parser.add_argument('--save_interval', nargs='?', type=int, default='10')
    parser.add_argument('--model_type', nargs='?', type=str, default='dispnetc')
    parser.add_argument('--maxdisp', type=int, default=192)

    # hyper params
    parser.add_argument('--lambda_cycle', type=float, default=10)
    parser.add_argument('--alpha_ssim', type=float, default=0.85)
    

    # load & save checkpoints
    parser.add_argument('--load_checkpoints', nargs='?', type=int, default=0, help='load from ckp(saved by Pytorch)')
    parser.add_argument('--load_from_mgpus_model', nargs='?', type=int, default=0, help='load ckp which is saved by mgus(nn.DataParallel)')
    parser.add_argument('--load_dispnet_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
    parser.add_argument('--checkpoint_save_path', nargs='?', type=str, default='checkpoints/best_checkpoint.pth.tar')

    # tensorboard, print freq
    parser.add_argument('--writer', nargs='?', type=str, default='StereoGAN')
    parser.add_argument('--print_freq', '-p', default=150, type=int, metavar='N', help='print frequency (default: 150)')

    parser.add_argument('--left_right_consistency', type = float, default = 0)

    # other
    parser.add_argument('--use_multi_gpu', nargs='?', type=int, default=0, help='the number of multi gpu to use')
    args = parser.parse_args()
    train(args)