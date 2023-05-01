import torch.nn.functional as F


def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.5, 0.7, 1.0, 1.3]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def corr_loss(disp_ests, disp_gt):
    weights = [1.0, 0.5]  #对应finetune和pred3
    all_losses = []
    #import pdb; pdb.set_trace()
    for i in range(len(disp_ests)):
        all_losses.append(weights[i] * F.smooth_l1_loss(disp_ests[i][0], disp_gt[i][0], size_average=True))
    return sum(all_losses)


