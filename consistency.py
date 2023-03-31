import torch
import torch.nn.functional as F
def apply_disparity(img, disp):
    #import pdb; pdb.set_trace()
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.arange(0, width, device=img.device).view(1, -1).repeat(height, 1)  # [256, 512]
    y_base = torch.arange(0, height, device=img.device).view(-1, 1).repeat(1, width)  # [256, 512]
    x_base = x_base.view(1, height, width).repeat(batch_size, 1, 1)  # [2, 256, 512]
    y_base = y_base.view(1, height, width).repeat(batch_size, 1, 1)  # [2, 256, 512]
    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    flow_field[:, :, :, 0] = 2.0 * flow_field[:, :, :, 0].clone() / max(width - 1, 1) - 1.0
    flow_field[:, :, :, 1] = 2.0 * flow_field[:, :, :, 1].clone() / max(height - 1, 1) - 1.0
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, flow_field, mode='bilinear', padding_mode='zeros')
    return output

def generate_image_left(img, disp):
    return apply_disparity(img, -disp)

def generate_image_right(img, disp):
    return apply_disparity(img, disp)