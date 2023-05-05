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
from models.bilinear_sampler import downsample_optical_flow, upsample_optical_flow
import flownet
from flownet.FlowNetC import FlowNetC
from flownet.FlowNetS import FlowNetS
from flownet.loss import multiscaleEPE



def val_flow(valloader, net_flow, writer, epoch=1, board_save=True):
    net_flow.eval()
    epe, f1_all = 0, 0
    epe1, fl_all1 = 0,0
    out_list, epe_list = [], [] 
    #out1_list, epe1_list = [],[]
    total_error = 0
    fl_error = 0
    results = {}
    average_over_pixels = False
    i = 0
    for left, right, disp_gt, left_forward,flow_gt,valid_gt in valloader:
        left_img = left.cuda()
        right_img = right.cuda()
        i = i + 1
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        left_forward = left_forward.cuda()
        flow_gt = flow_gt.cuda()
        valid_gt = valid_gt.cuda()
        
        input = [left_img,left_forward]
        input = torch.cat(input,1).to(device)
        #import pdb; pdb.set_trace()
        flow_preds = net_flow(input)
        flow_preds = [flow_preds]
        flow_preds_sampled = upsample_optical_flow(flow_preds,sample_factor=4, num_upsample=1)
        # useful when using parallel branches
        #flow_pr = results_dict['flow_preds']

        #flow = padder.unpad(flow_pr[0]).cpu()
        flow = flow_preds_sampled[0]
        #尺寸不对，还得上采样

        #epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        #mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
        #import pdb; pdb.set_trace()
        # flow1 = flow.squeeze(0)
        # flow_gt1 = flow_gt.squeeze(0)
        # epe1 =torch.sum((flow1 - flow_gt1) ** 2,dim=0).sqrt()
        # mag1 = torch.sum(flow_gt1 ** 2, dim = 0).sqrt()
        #print(epe1.shape,mag1.shape)
        epe = ((flow - flow_gt) ** 2).sqrt()
        mag = (flow_gt ** 2).sqrt()
        epe = epe.permute(0,2,3,1)
        mag = mag.permute(0,2,3,1)
        # epe1 = epe1.view(-1)
        # mag1 = mag1.view(-1)
        # val1 = valid_gt.view(-1) >= 0.5
        
        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        val = valid_gt >= 0.5
        #out1 = ((epe1 > 3.0) & ((epe1 / mag1) > 0.05)).float()
        
        # epe1_list.append(epe1[val1].mean().item())
        # out1_list.append(out1[val1].cpu().numpy())
        #print(epe.shape,mag.shape,val.shape,out.shape)
        if average_over_pixels:
            epe_list.append(epe[val].cpu().numpy())
        else:
            epe_list.append(epe[val].mean().item())
        #import pdb; pdb.set_trace()
        out_list.append(out[val].cpu().numpy())
        '''
        mask = np.ceil(np.clip(np.abs(flow_gt[0,0].cpu().numpy()), 0, 1))
        epe1, f1 = evaluate_flow(flow.cpu().numpy(),flow_gt.cpu().numpy())
        total_error += epe1
        fl_error += f1
        '''
        loss_flow, metrics = flow_loss_func_val(flow, flow_gt, valid_gt,
                                            gamma=args.gamma,
                                            max_flow=args.max_flow,
                                            )
        epe1 = epe1+metrics['epe']
        fl_all1 = fl_all1 + metrics['fl_all']
    '''   
    total_error /= i
    fl_error /= i
    '''
    if average_over_pixels:
        epe_list = np.concatenate(epe_list)
    else:
        epe_list = np.array(epe_list)
    #import pdb; pdb.set_trace()
    out_list = np.concatenate(out_list)
    #out1_list = np.concatenate(out1_list)
    epe = np.mean(epe_list)
    
    # epe1_list = np.array(epe1_list)
    # epe1 = np.mean(epe1_list)
    f1_all = np.mean(out_list)
    # f1_all1 = 100 * np.mean(out1_list)

    #import pdb; pdb.set_trace()
    epe1 = epe1/i
    fl_all1 = fl_all1/i
    if board_save:
        writer.add_scalar("val/EPE_flow", epe, epoch)
        writer.add_scalar("val/Fl_all", f1_all, epoch)
        writer.add_scalar("val/EPE_flow1", epe, epoch)
        writer.add_scalar("val/Fl_all1", f1_all, epoch)

    print('epe:', epe, 'f1_all',f1_all)
    print('epe1:', epe1,'fl_all1',fl_all1)
    
    
    return epe, f1_all, epe1, fl_all1

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
    
    # add optical flow module
    
    model_names = sorted(name for name in flownet.__dict__
                    if name.islower() and not name.startswith("__"))
    #import pdb; pdb.set_trace()
    #from flownet import FlowNetC
    #net_flow = FlowNetC(batchNorm=False)
    net_flow = FlowNetS(batchNorm=False)
    #net_flow = flownet.__dict__['flownetc']

    if args.load_checkpoints:
        if args.load_from_mgpus_model:
            if args.load_dispnet_path:
                net = load_multi_gpu_checkpoint(net, args.load_dispnet_path, 'model')
            else:
                net.apply(weights_init_normal)

            if args.load_flownet_path:
                net_flow = load_multi_gpu_checkpoint(net_flow,args.load_flownet_path,'model_flow')
            else:
                net_flow.apply(weights_init_normal) #初始化?

            # G_A_forward = load_multi_gpu_checkpoint(G_A_forward, args.load_gan_path, 'G_A_forward')
            # G_A_backward = load_multi_gpu_checkpoint(G_A_backward, args.load_gan_path, 'G_A_backward')
            # D_A_forward = load_multi_gpu_checkpoint(D_A_forward, args.load_gan_path,'D_A_forward')
            # D_A_backward = load_multi_gpu_checkpoint(D_A_backward,args.load_gan_path,'D_A_backward')

        else:
            if args.load_dispnet_path:
                net = load_checkpoint(net, args.load_checkpoint_path, device)
            else:
                net.apply(weights_init_normal)
            if args.load_flownet_path:
                net_flow = load_checkpoint(net_flow, args.load_checkpoint_path, device)
            else:
                net_flow.apply(weights_init_normal)   #可能有问题？
        

    else:
        
        net_flow.apply(weights_init_normal)
    

    # optimizer = optim.SGD(params, momentum=0.9)
    #加载optimizer的
    
    #if args.flow:
    optimizer_flow = optim.Adam(net_flow.parameters(),lr=args.lr_flow,weight_decay = args.weight_decay)
        
    # start epoch赋初值
    start_epoch = 0
    if args.load_checkpoints:
        print('load optimizer')
        checkpoint = torch.load(args.load_flownet_path,map_location = device)
        optimizer_flow.load_state_dict(checkpoint['optimizer_flow_state_dict'])
        for state in optimizer_flow.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        
    if args.use_multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        if args.flow:
            net_flow = nn.DataParallel(net_flow, device_ids=list(range(args.use_multi_gpu)))
            
    if args.flow:
        net_flow.to(device)

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
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16,pin_memory=True,prefetch_factor=4)
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
            
                
            # train disp net
            
            if args.flow:
                net_flow.train()
                optimizer_flow.zero_grad()
            # import IPython
            # IPython.embed()
                
            # add optical flow: flow的image输入输出是否和stereo matching 任务一样？

            #left_A_forward用下一帧的右图生成
            #left_A_forward = G()
            flowA = flowA.permute(0,3,1,2)
            #print('flowA.shape:',flowA.shape)
            #print('flow_preds.shape:',flow_preds[0].shape, len(flow_preds))
            #print(flowA[0,0].shape, flowA[0,1].shape)
            if args.source_dataset == 'driving':
                validA = (flowA[:,0,:,:].abs() < 1000) & (flowA[:,1,:,:].abs() < 1000)
            elif args.source_dataset == 'VKITTI2':                     #VKITTI2
                validA = validA
            else:
                print('dataset name error!')
                raise "No suportive dataset"

            validA = validA.float()  #解决这里
            #print('validA.shape',validA.shape)
            #print(G_AB(leftA).shape, G_AB.forward(leftA_forward).shape)
            img1 = leftA
            img2 = leftA_forward
            input = [img1,img2]
            input = torch.cat(input,1).to(device)
            flow_preds = net_flow(input)
            #import pdb; pdb.set_trace()
                                                    # flow_gt shape:[1,2,320,1152] valid shape:[1,320,1152]
            loss_flow = multiscaleEPE(flow_preds, flowA, weights=args.multiscale_weights, sparse=args.sparse)
        
            loss = loss_flow
            #print(loss)
            loss.backward()
            
            optimizer_flow.step()

            if i % print_freq == print_freq - 1:
                print('epoch[{}/{}]  step[{}/{}]  loss: {}'.format(epoch, args.total_epochs, i, len(trainloader), loss.item() ))
                train_loss_meter.update(running_loss / print_freq)
                
                writer.add_scalar('loss/loss_flow', loss_flow, train_loss_meter.count * print_freq)
                

        # TODO: 加 optical flow的evaluation metrics,看能不能相互促进
        with torch.no_grad():
            
            epe_flow, f1_all, epe1_flow, fl_all1 = val_flow(valloader, net_flow, writer, epoch=epoch, board_save=True)

        t1 = time.time()   #to do: add other evaluation metrics
        
        print('epoch:{}, epe_flow:{:.4f}, f1_all:{:.4f},epe1_flow:{:.4f}, fl_all1:{:.4f}, cost time:{} '.format(epoch, epe_flow,f1_all, epe1_flow, fl_all1, t1-t))
        if (epoch % args.save_interval == 0) or f1_all < best_val_f1 or epe_flow < best_val_epe or epe1_flow < best_val_epe1:
            best_val_f1 = f1_all
            best_val_epe = epe_flow
            best_val_epe1 = epe1_flow
            # add flow
            if args.flow:
                torch.save({
                            'epoch': epoch,
                            'model_flow':net_flow.state_dict(),
                            'optimizer_flow_state_dict':optimizer_flow.state_dict(),
                            }, args.checkpoint_save_path + '/ep' + str(epoch) + '_epe_flow{:.4f}_f1_all{:.4f}_epe1_flow{:.4f}_fl_all1{:.4f}'.format(epe_flow,f1_all, epe1_flow, fl_all1) + '.pth')

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
    
    #flownet parameters
    parser.add_argument('--flow', type = float,help= 'add optical flow branch', default = 1)

    parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
    parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                    'automatically seleted when choosing a KITTIdataset')
    
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='flownets',
    #                    choices=model_names,
    #                    help='model architecture, overwritten if pretrained is specified: ' +
    #                   ' | '.join(model_names))
    
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='exponential weighting')
    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--lr_flow', nargs='?', type=float, default=1e-4, help='learning rate for unimatch flow')
    parser.add_argument('--weight_decay', default=4e-4, type=float)

    # use unimatch stereo part
    parser.add_argument('--unimatch_stereo',type = float,help= 'use unimatch stereo as dispnet', default = 1)
    # parser.add_argument('--max_disp', default=400, type=int,
    #                     help='exclude very large disparity in the loss function')
    parser.add_argument('--debug', type=float, default=1)
    args = parser.parse_args()

    torch.manual_seed(3407)
    np.random.seed(3407)
    
    train(args)
