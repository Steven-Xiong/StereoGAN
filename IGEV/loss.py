import torch
import torch.nn.functional as F

def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()


    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], size_average=True)
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return disp_loss, metrics
    
#增加correlation loss
def corr_loss(disp_preds, disp_init_pred, disp_gt, disp_gt_init, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """
    
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    #mag = torch.sum(disp_gt**2, dim=1).sqrt()
    #valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    #assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    #assert not torch.isinf(disp_gt[valid.bool()]).any()

    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred, disp_gt_init, size_average=True)
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt[i]).abs()
        #assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt[i].shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss.mean()

    # epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    # epe = epe.view(-1)[valid.view(-1)]

    return disp_loss