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
from IGEV.igev_stereo import IGEVStereo
from IGEV.loss import sequence_loss, corr_loss

def val(valloader, net, writer,iters, epoch=1, board_save=True):
    net.eval()
    EPEs, D1s, Thres1s, Thres2s, Thres3s = 0, 0, 0, 0, 0
    i = 0
    for left_img, right_img, disp_gt, left_forward, flow, valid in valloader:
        left_img = left_img.cuda()
        right_img = right_img.cuda()
        disp_gt = disp_gt.cuda()
        i = i + 1
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0) #有的是0.5？
        if args.IGEV:
            disp_est = net(left_img, right_img, iters=iters, test_mode=True).squeeze(1)  #取最后一维度？
        else:
            disp_est = net(left_img, right_img)[0].squeeze(1)
        #import pdb; pdb.set_trace()
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

def evaluate_flow(flow, flow_gt, valid_mask=None):
    import pdb; pdb.set_trace()
    if valid_mask is None:
        tmp = np.multiply(flow_gt[:, 0,:,:], flow_gt[:, 1,:,:])
        valid_mask = np.ceil(np.clip(np.abs(tmp), 0, 1))
        
    N = np.sum(valid_mask)

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    u_gt = flow_gt[:, 0, :, :]
    v_gt = flow_gt[:, 1, :, :]

    ### compute_EPE
    du = u - u_gt
    dv = v - v_gt

    du2 = np.multiply(du, du)
    dv2 = np.multiply(dv, dv)

    EPE = np.multiply(np.sqrt(du2 + dv2), valid_mask)
    EPE_avg = np.sum(EPE) / N
    
    ### compute FL
    bad_pixels = np.logical_and(
        EPE > 3,
        (EPE / np.sqrt(np.sum(np.square(flow_gt), axis=0)) + 1e-5) > 0.05)
    FL_avg = bad_pixels.sum() / valid_mask.sum()

    return EPE_avg, FL_avg

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
        
        results_dict = net_flow(left, left_forward,
                                 attn_type=args.attn_type,
                                 attn_splits_list=args.attn_splits_list,
                                 corr_radius_list=args.corr_radius_list,
                                 prop_radius_list=args.prop_radius_list,
                                 num_reg_refine=args.num_reg_refine,
                                 task='flow',
                             )
        
        # useful when using parallel branches
        flow_pr = results_dict['flow_preds']

        #flow = padder.unpad(flow_pr[0]).cpu()
        flow = flow_pr[-1]
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
        loss_flow, metrics = flow_loss_func_val(flow_pr, flow_gt, valid_gt,
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

            if args.load_flownet_path:
                net_flow = load_multi_gpu_checkpoint(net_flow,args.load_flownet_path,'model_flow')
            else:
                net_flow.apply(weights_init_normal)

        else:
            if args.load_IGEV_path:
                net = load_checkpoint(net, args.load_checkpoint_path, device)
            else:
                net.apply(weights_init_normal)
            if args.load_flownet_path:
                net_flow = load_checkpoint(net_flow, args.load_checkpoint_path, device)
            else:
                net_flow.apply(weights_init_normal)   #可能有问题？
            

    else:
        #net.apply(weights_init_normal)
        
        if args.flow:
            net_flow.apply(weights_init_normal)
           
    # optimizer = optim.SGD(params, momentum=0.9)
    #加载optimizer的
    
    if args.IGEV:
        optimizer = optim.AdamW(net.parameters(), lr=args.lr_rate,  #调一样不要整混
                                weight_decay=args.weight_decay_IGEV, eps=1e-8)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr_rate, betas=(0.9, 0.999))
    
    if args.flow:
        #optimizer_flow = optim.Adam(net_flow.parameters(), lr=args.lr_flow, betas=(0.9, 0.999))
        optimizer_flow = optim.AdamW(net_flow.parameters(),lr=args.lr_flow,weight_decay = args.weight_decay)
        # optimizer_G_flow = optim.Adam(itertools.chain(G_A_forward.parameters(), G_A_backward.parameters()), lr=args.lr_gan, betas=(0.5, 0.999))
        # optimizer_D_forward = optim.Adam(D_A_forward.parameters(), lr=args.lr_gan, betas=(0.5, 0.999))
        # optimizer_D_backward = optim.Adam(D_A_backward.parameters(), lr=args.lr_gan, betas=(0.5, 0.999))
    # start epoch赋初值
    start_epoch = 0
    start_step=0

    if args.load_checkpoints:
        print('load optimizer')
        checkpoint = torch.load(args.load_IGEV_path,map_location = device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        start_step = checkpoint['step']
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        if args.flow:
            optimizer_flow.load_state_dict(checkpoint['optimizer_flow_state_dict'])
            for state in optimizer_flow.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    
    if args.use_multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=list(range(args.use_multi_gpu)))
        if args.flow:
            net_flow = nn.DataParallel(net_flow, device_ids=list(range(args.use_multi_gpu)))
            
    #add scheduler：
    last_epoch = start_step if args.load_checkpoints and start_step > 0 else -1
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr_rate, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear',last_epoch=last_epoch)
    total_steps = start_step

    net.to(device)
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
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    valdataset = ValJointImageDataset()
    valloader = torch.utils.data.DataLoader(valdataset, batch_size=args.test_batch_size, shuffle=False, num_workers=16)

    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    ## debug only
    #with torch.no_grad():
    #    l1_test_loss, out_val = val(valloader, net, G_AB, None, writer, epoch=0, board_save=True)
    #    val_loss_meter.update(l1_test_loss)
    #    print('Val epoch[{}/{}] loss: {}'.format(0, args.total_epochs, l1_test_loss))

    print('begin training...')
    print('start_epoch:', start_epoch)
    print('start_step:', start_step)
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
        if args.IGEV:
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
                validA = batch['validA'].to(device)  #VKITTI2直接有, driving为空
            validB = batch['validB'].to(device)
            out_shape = (leftA.size(0), 1, args.img_height//16, args.img_width//16)
            valid = torch.cuda.FloatTensor(np.ones(out_shape))
            fake = torch.cuda.FloatTensor(np.zeros(out_shape))
            
                
            # train disp net
            net.train()
            if args.flow:
                net_flow.eval()
            if args.IGEV:
                net.module.freeze_bn() # We keep BatchNorm frozen
            
            optimizer.zero_grad()
            
            # import IPython
            # IPython.embed()
            
            loss_disp = 0
            if args.IGEV:
                # loss_disp = 0
                disp_init_pred,pred_disps = net(leftA, rightA,iters = args.train_iters)
                #mport pdb; pdb.set_trace()

                mask = (dispA < args.maxdisp) & (dispA > 0)
                mask = mask.squeeze(1)
                loss_disp, metrics = sequence_loss(pred_disps, disp_init_pred, dispA, mask, max_disp=args.max_disp)
                loss0 = loss_disp
                # loss weights
                
                #print(len(pred_disps))
                #print(pred_disps[0].shape, pred_disps[1].shape,pred_disps[2].shape,pred_disps[-1].shape)

                
            else:
                import pdb; pdb.set_trace()
                disp_ests = net(leftA,rightA)  #各种feature, len=7
        
                # 加自己的predA_disp跟predB_disp之间的loss
                # if args.result_adv:
                #     disp_estsB = net(leftB,rightB)
                
                #print(len(disp_ests))
                #print(disp_ests[0].shape,disp_ests[1].shape,disp_ests[2].shape,disp_ests[3].shape)
                #print('disp_ests[0]:',disp_ests[0].squeeze(1).shape)
                mask = (dispA < args.maxdisp) & (dispA > 0)
                #print(disp_ests.shape, dispA.shape)
                #import pdb; pdb.set_trace()
                loss0 = model_loss0(disp_ests, dispA, mask)

            total_steps +=1
            loss = loss0
            loss.backward()
            optimizer.step()
            scheduler.step()
            # add optical flow: flow的image输入输出是否和stereo matching 任务一样？

            #left_A_forward用下一帧的右图生成
            #left_A_forward = G()
            
            loss_flow=0
            if args.flow:
                net_flow.train()
                net.eval()
                optimizer_flow.zero_grad()
            if args.flow:
                #print(G_AB(leftA).shape, G_AB.forward(leftA_forward).shape)
                results_dict = net_flow(leftA, leftA_forward,
                                 attn_type=args.attn_type,
                                 attn_splits_list=args.attn_splits_list,
                                 corr_radius_list=args.corr_radius_list,
                                 prop_radius_list=args.prop_radius_list,
                                 num_reg_refine=args.num_reg_refine,
                                 task='flow',
                                 )
                #import pdb; pdb.set_trace()
                flow_preds = results_dict['flow_preds']   #这里存flow pred模型
                                                        # flow_gt shape:[1,2,320,1152] valid shape:[1,320,1152]
                
                
                flowA = flowA.permute(0,3,1,2)
                #print('flowA.shape:',flowA.shape)
                #print('flow_preds.shape:',flow_preds[0].shape, len(flow_preds))
                #print(flowA[0,0].shape, flowA[0,1].shape)
                
                if args.source_dataset == 'driving':
                    validA = (flowA[:,0,:,:].abs() < 1000) & (flowA[:,1,:,:].abs() < 1000)
                elif args.source_dataset == 'VKITTI2':
                    validA = validA
                else:
                    print('dataset name error!')
                    raise "No suportive dataset"

                validA = validA.float()  #解决这里
                #print('validA.shape',validA.shape)
                loss_flow, metrics = flow_loss_func(flow_preds, flowA, validA,
                                            gamma=args.gamma,
                                            max_flow=args.max_flow,
                                            )
            else:
                loss_flow, metrics= 0, 0
            
            
            if args.flow:
                loss_flow.backward()
                optimizer_flow.step()

            if i % print_freq == print_freq - 1:

                if args.flow:
                    print('epoch[{}/{}]  step[{}/{}]  loss: {} loss_flow: {}'.format(epoch, args.total_epochs, i, len(trainloader), loss.item(), loss_flow.item() ))
                else:
                    print('epoch[{}/{}]  step[{}/{}]  loss: {} '.format(epoch, args.total_epochs, i, len(trainloader), loss.item() ))
                train_loss_meter.update(running_loss / print_freq)
                #writer.add_scalar('loss/trainloss avg_meter', train_loss_meter.val, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_disp', loss, train_loss_meter.count * print_freq)
                if args.flow:
                    writer.add_scalar('loss/loss_flow', loss_flow, train_loss_meter.count * print_freq)
                
        # TODO: 加 optical flow的evaluation metrics,看能不能相互促进
        with torch.no_grad():
            EPE, D1,Thres1s,Thres2s,Thres3s = val(valloader, net, writer,iters = args.val_iters, epoch=epoch, board_save=True)
            if args.flow:
                epe_flow, f1_all, epe1_flow, fl_all1 = val_flow(valloader, net_flow, writer, epoch=epoch, board_save=True)

        t1 = time.time()   #to do: add other evaluation metrics
        print('epoch:{}, D1:{:.4f}, EPE:{:.4f},Thres2s:{:.4f},Thres4s:{:.4f},Thres5s:{:.4f}, cost time:{} '.format(epoch, D1, EPE,Thres1s,Thres2s,Thres3s, t1-t))
        # add flow
        if args.flow:
            print('epoch:{}, epe_flow:{:.4f}, f1_all:{:.4f},epe1_flow:{:.4f}, fl_all1:{:.4f}, cost time:{} '.format(epoch, epe_flow,f1_all, epe1_flow, fl_all1, t1-t))
        if (epoch % args.save_interval == 0) or D1 < best_val_d1 or EPE < best_val_epe:
            best_val_d1 = D1
            best_val_epe = EPE
            # add flow
            if args.flow:
                best_val_epe_flow = epe_flow
                best_f1_all = f1_all
            if args.flow:
                torch.save({
                            'epoch': epoch,
                            'step': total_steps,
                            'model': net.state_dict(),
                            'model_flow':net_flow.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'optimizer_flow_state_dict':optimizer_flow.state_dict(),
                            }, args.checkpoint_save_path + '/ep' + str(epoch) + '_D1_{:.4f}_EPE{:.4f}_Thres2s{:.4f}_Thres4s{:.4f}_Thres5s{:.4f}_epe_flow{:.4f}_f1_all{:.4f}_epe1_flow{:.4f}_fl_all1{:.4f}'.format(D1, EPE,Thres1s,Thres2s,Thres3s,epe_flow,f1_all, epe1_flow, fl_all1) + '.pth')
            
            
            else:
                torch.save({
                        'epoch': epoch,
                        'step': total_steps,
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
    parser.add_argument('--lrepochs', type=str, default='30:1', help='the epochs to  lr: the downscale rate')
    parser.add_argument('--lr_gan', nargs='?', type=float, default=2e-4, help='learning rate for GAN')
    parser.add_argument('--train_ratio_gan', nargs='?', type=int, default=5, help='training ratio disp:gan=5:1')
    parser.add_argument('--batch_size', nargs='?', type=int, default=6, help='batch size')
    parser.add_argument('--test_batch_size', nargs='?', type=int, default=4, help='test batch size')
    parser.add_argument('--total_epochs', nargs='?', type=int, default='201')
    parser.add_argument('--save_interval', nargs='?', type=int, default='10')
    parser.add_argument('--model_type', nargs='?', type=str, default='dispnetc')
    parser.add_argument('--maxdisp', type=int, default=192)
    # add steps:
    parser.add_argument('--num_steps', type = int, default=200000)

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
    parser.add_argument('--load_IGEV_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
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

    # use igev part
    parser.add_argument('--IGEV',type = float,help= 'use unimatch stereo as dispnet', default = 1)
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--val_iters', type=int, default=32, help="number of updates to the disparity field in each forward pass.")
    # parser.add_argument('--max_disp', default=400, type=int,
    #                     help='exclude very large disparity in the loss function')
    parser.add_argument('--debug', type=float, default=1)
    parser.add_argument('--weight_decay_IGEV', type = float,default = 0.00001)
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

    args = parser.parse_args()
    torch.manual_seed(3407)
    np.random.seed(3407)
    train(args)
