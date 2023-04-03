if args.flow:
    G_forward.train()
    G_backward.train()
    net_flow.eval()
    optimizer_G_flow.zero_grad()
    # train generators

    # Identity loss
    loss_id_forward = (criterion_identity(G_backward(leftA), leftA) + criterion_identity(G_backward(rightA), rightA)) / 2
    loss_id_backward = (criterion_identity(G_forward(leftA_forward), leftA_forward) + criterion_identity(G_forward(rightB), rightB)) / 2
    loss_id = (loss_id_forward + loss_id_backward) / 2

    if args.lambda_warp_inv:
        fake_leftA_forward, fake_leftA_forward_feats = G_forward(leftA, extract_feat=True)
        fake_leftA, fake_leftA_feats = G_backward(leftA_forward, extract_feat=True)
    else:
        fake_leftA_forward = G_forward(leftA)
        fake_leftA = G_backward(leftA_forward)
    if args.lambda_warp:
        fake_rightB, fake_rightB_feats = G_forward(rightA, extract_feat=True)
        fake_rightA, fake_rightA_feats = G_backward(rightB, extract_feat=True)
    else:
        fake_rightB = G_forward(rightA)
        fake_rightA = G_backward(rightB)
    loss_GAN_AB = criterion_GAN(D_B(fake_leftA_forward), valid)
    loss_GAN_BA = criterion_GAN(D_A(fake_leftA), valid)
    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

    if args.lambda_warp_inv:
        rec_leftA, rec_leftA_feats = G_backward(fake_leftA_forward, extract_feat=True)
    else:
        rec_leftA = G_backward(fake_leftA_forward)
    if args.lambda_warp:
        rec_rightA, rec_rightA_feats = G_backward(fake_rightB, extract_feat=True)
    else:
        rec_rightA = G_backward(fake_rightB)
    rec_leftA_forward = G_forward(fake_leftA)
    rec_rightB = G_forward(fake_rightA)
    loss_cycle_A = (criterion_identity(rec_leftA, leftA) + criterion_identity(rec_rightA, rightA)) / 2
    loss_ssim_A = 1. - (ssim_loss(rec_leftA, leftA) + ssim_loss(rec_rightA, rightA)) / 2
    loss_cycle_B = (criterion_identity(rec_leftA_forward, leftA_forward) + criterion_identity(rec_rightB, rightB)) / 2
    loss_ssim_B = 1. - (ssim_loss(rec_leftA_forward, leftA_forward) + ssim_loss(rec_rightB, rightB)) / 2
    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
    loss_ssim = (loss_ssim_A + loss_ssim_B) / 2
    #print('loss_ssim', loss_ssim)
    # add cosine similarity loss
    #import pdb; pdb.set_trace()
    if args.cosine_similarity:
        #print('rec_leftA',rec_leftA.shape)
        #print('leftA',leftA.shape)
        loss_cosineA = 1- (F.cosine_similarity(rec_leftA, leftA,dim=-1).mean() + F.cosine_similarity(rec_leftA, leftA,dim=-1).mean()) / 2
        loss_cosineB = 1- (F.cosine_similarity(rec_leftA_forward, leftA_forward,dim=-1).mean() + F.cosine_similarity(rec_leftA_forward, leftA_forward,dim=-1).mean()) / 2
        loss_cosine = (loss_cosineA + loss_cosineB) /2
        #print(loss_cosine)
    else:
        loss_cosine = 0
    # add perceptual loss:
    if args.perceptual:
        loss_perceptualA = criterion_perceptual(rec_leftA, leftA).mean() 
        loss_perceptualB = criterion_perceptual(rec_leftA_forward, leftA_forward).mean() 
        loss_perceptual = (loss_perceptualA + loss_perceptualB)/2
    else:
        loss_perceptual = 0 

    # mode seeking loss
    if args.lambda_ms:
        loss_ms = G_forward(leftA, zx=True, zx_relax=True).mean()
    else:
        loss_ms = 0

    # warping loss
    if args.lambda_warp_inv:
        fake_leftA_forward_warp, loss_warp_inv_feat1 = G_forward(rightA, -dispA, True, [x.detach() for x in fake_leftA_forward_feats])
        rec_leftA_warp, loss_warp_inv_feat2 = G_backward(fake_rightB, -dispA, True, [x.detach() for x in rec_leftA_feats])
        loss_warp_inv1 = warp_loss([(G_backward(fake_leftA_forward_warp[0]), fake_leftA_forward_warp[1])], [leftA], weights=[1])
        loss_warp_inv2 = warp_loss([rec_leftA_warp], [leftA], weights=[1])
        loss_warp_inv = loss_warp_inv1 + loss_warp_inv2 + loss_warp_inv_feat1.mean() + loss_warp_inv_feat2.mean()
    else:
        loss_warp_inv = 0

    if args.lambda_warp:
        fake_rightB_warp, loss_warp_feat1 = G_forward(leftA, dispA, True, [x.detach() for x in fake_rightB_feats])
        rec_rightA_warp, loss_warp_feat2 = G_backward(fake_leftA_forward, dispA, True, [x.detach() for x in rec_rightA_feats])
        loss_warp1 = warp_loss([(G_backward(fake_rightB_warp[0]), fake_rightB_warp[1])], [rightA], weights=[1])
        loss_warp2 = warp_loss([rec_rightA_warp], [rightA], weights=[1])
        loss_warp = loss_warp1 + loss_warp2 + loss_warp_feat1.mean() + loss_warp_feat2.mean()
    else:
        loss_warp = 0

    # corr loss
    if args.lambda_corr:
        
        corrB = net(leftA_forward, rightB, extract_feat=True)
        #print(corrB[0].shape,corrB[1].shape)
        
        corrB1 = net(leftA_forward, rec_rightB, extract_feat=True)
        corrB2 = net(rec_leftA_forward, rightB, extract_feat=True)
        corrB3 = net(rec_leftA_forward, rec_rightB, extract_feat=True)
        
        #import pdb; pdb.set_trace()
        loss_corr = (criterion_identity(corrB1, corrB)+criterion_identity(corrB2, corrB)+criterion_identity(corrB3, corrB))/3
    else:
        loss_corr = 0.

    lambda_ms = args.lambda_ms * (args.total_epochs - epoch) / args.total_epochs
    loss_G = loss_GAN + args.lambda_cycle*(args.alpha_ssim*loss_ssim+(1-args.alpha_ssim)*loss_cycle) + args.lambda_id*loss_id \
        + args.lambda_warp*loss_warp + args.lambda_warp_inv*loss_warp_inv + args.lambda_corr*loss_corr + lambda_ms*loss_ms \
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
    loss_real_B = criterion_GAN(D_B(leftA_forward), valid)
    fake_leftA_forward.detach_()
    loss_fake_B = criterion_GAN(D_B(fake_leftA_forward), fake)
    loss_D_B = (loss_real_B + loss_fake_B) / 2
    loss_D_B.backward()
    optimizer_D_B.step()