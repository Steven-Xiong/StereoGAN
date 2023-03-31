import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision

def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def model_loss0(disp_ests, disp_gt, mask):
    scale = [0, 1, 2, 3, 4, 5, 6]
    weights = [1, 1, 1, 0.8, 0.6, 0.4, 0.2]
    all_losses = []
    #import pdb; pdb.set_trace()
    for disp_est, weight, s in zip(disp_ests, weights, scale):
        #print(disp_est.shape)
        if s != 0:
            dgt = F.interpolate(disp_gt, scale_factor=1/(2**s))
            m = F.interpolate(mask.float(), scale_factor=1/(2**s)).byte()
        else:
            dgt = disp_gt
            m = mask
        all_losses.append(weight * F.smooth_l1_loss(disp_est[m], dgt[m], size_average=True))
    return sum(all_losses)

def warp_loss(gen, real, weights=[0.5,0.5,0.7]):
    #weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for g0, r, weight in zip(gen, real, weights):
        g, m = g0
        m = m.float()
        #perm = torch.randperm(g.size(1))[:3]
        #m = (g[:,perm,:,:].abs() > 1e-4)
        #all_losses.append(weight * F.l1_loss(g[:,perm,:,:][m], r[:,perm,:,:][m], size_average=True))
        all_losses.append(weight * (m * F.l1_loss(g, r, reduction='none').mean(1)).mean())
    return sum(all_losses)

# 加perceptual loss

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

class PerceptualLoss(nn.Module):
    def __init__(self, normalize_inputs=True):
        super(PerceptualLoss, self).__init__()
        self.normalize_inputs = normalize_inputs

        vgg = torchvision.models.vgg19(pretrained=True).features
        self.mean_ = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None] #IMAGENET_MEAN
        self.std_ = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None] #IMAGENET_STD

        vgg_avg_pooling = []

        for weights in vgg.parameters():
            weights.requires_grad = False

        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)

        self.vgg = nn.Sequential(*vgg_avg_pooling)

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def partial_losses(self, input, target):
        input = (input + 1) / 2
        target = (target.detach() + 1) / 2
        losses = []

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
            features_target = self.do_normalize_inputs(target)
        else:
            features_input = input
            features_target = target

        for layer in self.vgg[:30]:

            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                loss = F.l1_loss(features_input, features_target, reduction='none')
                loss = loss.mean(dim=tuple(range(1, len(loss.shape))))
                losses.append(loss)

        return losses

    def forward(self, input, target):
        losses = self.partial_losses(input, target)
        return torch.stack(losses).sum(dim=0)

    def get_global_features(self, input):
        input = (input + 1) / 2
        losses = []

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
        else:
            features_input = input

        features_input = self.vgg(features_input)
        return features_input

#edge aware smoothness loss
def smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

# borrow from PAM
def loss_disp_unsupervised(img_left, img_right, disp, valid_mask=None, mask=None):
    b, _, h, w = img_left.shape
    image_warped = warp_disp(img_right, -disp)

    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask

    loss = 0.15 * L1Loss(image_warped * valid_mask, img_left * valid_mask) + \
           0.85 * (valid_mask * (1 - ssim(img_left, image_warped)) / 2).mean()
    return loss

def warp_disp(img, disp):
    '''
    Borrowed from: https://github.com/OniroAI/MonoDepth-PyTorch
    '''
    b, _, h, w = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :] / w
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='border')

    return output