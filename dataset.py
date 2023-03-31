import glob
import random
import os
import png
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from utils.util import read_all_lines, pfm_imread

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

# crop to (256, 512), both left and right images, conditioned on disp
class ImageDataset(Dataset):
    def __init__(self, rootA='/home/autel/xzx/Data/feng/data/sceneflow/driving', rootB='/home/autel/xzx/Data/feng/data/kitti_15', height=320, width=512, transforms_=None,left_right_consistency=1):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.width = width
        self.height = height
        self.left_right_consistency = left_right_consistency
        transforms_ = [transforms.ToTensor(),
                       transforms.Normalize(mean=mean, std=std)]
        self.transform = transforms.Compose(transforms_)
        self.rootA = rootA
        self.rootB = rootB
        self.rootB_pseudo_label = '/home/autel/xzx/CREStereo/vis_results/CREStereo/data/pseudo_label_png'
        self.rootB_error_map = '/home/autel/xzx/CREStereo/vis_results/CREStereo/data/error_map_pfm'
        self.leftA_files, self.rightA_files, self.dispA_files = self.load_path('filenames/driving_adv.txt')
        self.leftB_files, self.rightB_files, self.dispB_files = self.load_path('filenames/kitti15_adv.txt')

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def load_dispA(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __getitem__(self, index):
        index2 = random.randint(0, len(self.leftB_files) - 1)
        leftA = self.load_image(os.path.join(self.rootA, self.leftA_files[index]))
        rightA = self.load_image(os.path.join(self.rootA, self.rightA_files[index]))
        dispA = self.load_dispA(os.path.join(self.rootA, self.dispA_files[index]))
        leftB = self.load_image(os.path.join(self.rootB, self.leftB_files[index2]))
        rightB = self.load_image(os.path.join(self.rootB, self.rightB_files[index2]))
        if self.left_right_consistency:
            dispB = self.load_disp(os.path.join(self.rootB_pseudo_label, self.dispB_files[index2].split('/')[-1]))
            error_mapB = self.load_dispA(os.path.join(self.rootB_error_map, self.dispB_files[index2].split('/')[-1].replace('.png','.pfm')))
        else:
            dispB = self.load_disp(os.path.join(self.rootB, self.dispB_files[index2]))
            error_mapB = []
        
        crop_w, crop_h = self.width, self.height
        wA, hA = leftA.size
        wB, hB = leftB.size
        x1 = random.randint(0, wA - crop_w)
        #y1 = random.randint(70, hA-70-crop_h)
        y1 = random.randint(0, hA - crop_h)
        x2 = random.randint(0, wB - crop_w)
        y2 = random.randint(0, hB - crop_h)

        # random crop
        leftA = leftA.crop((x1, y1, x1+crop_w, y1+crop_h))
        rightA = rightA.crop((x1, y1, x1+crop_w, y1+crop_h))
        dispA = dispA[y1:y1+crop_h, x1:x1+crop_w]
        leftB = leftB.crop((x2, y2, x2+crop_w, y2+crop_h))
        rightB = rightB.crop((x2, y2, x2+crop_w, y2+crop_h))
        dispB = dispB[y2:y2+crop_h, x2:x2+crop_w]
        if self.left_right_consistency:
            error_mapB = error_mapB[y2:y2+crop_h, x2:x2+crop_w]
        else:
            error_mapB = []
        # transform
        leftA = self.transform(leftA)
        rightA = self.transform(rightA)
        leftB = self.transform(leftB)
        rightB = self.transform(rightB)
        return {"leftA": leftA, "rightA": rightA, "dispA": dispA, "leftB": leftB, "rightB": rightB, "dispB": dispB, "error_mapB": error_mapB}

    def __len__(self):
        return len(self.leftA_files)

# for validation
class ValJointImageDataset(Dataset):
    def __init__(self, root='/home/autel/xzx/Data/feng/data/kitti_15', transforms_=None, input_shape=(3, 384, 1280)):
        f = open('./filenames/kitti15_adv.txt', 'r')
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        channels, height, width = input_shape
        transforms_ = [transforms.ToTensor(),
                       transforms.Normalize(mean=mean, std=std)]
        self.transform = transforms.Compose(transforms_)
        #transforms_disp = [#transforms.Resize((height, width), Image.BICUBIC),
        #                   transforms.ToTensor()]
        #self.transform_disp = transforms.Compose(transforms_disp)

        self.left_files = []
        self.right_files = []
        self.disp_files = []
        for line in f:
            line = line.strip()
            a, b, c = line.split()
            self.left_files.append(os.path.join(root, a))
            self.right_files.append(os.path.join(root, b))
            self.disp_files.append(os.path.join(root, c))

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __getitem__(self, index):
        left = self.load_image(self.left_files[index])
        shape = np.array(left).shape[:2]
        right = self.load_image(self.right_files[index])
        disp = self.load_disp(self.disp_files[index])
        #disp = Image.fromarray(disp)

        top_pad = 384 - shape[0]
        right_pad = 1280 - shape[1]
        assert top_pad > 0 and right_pad > 0
        # pad disparity gt
        if disp is not None:
            assert len(disp.shape) == 2
            disp = np.lib.pad(disp, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        left = self.transform(left).numpy()
        right = self.transform(right).numpy()
        left = np.lib.pad(left, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        right = np.lib.pad(right, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        #disp = self.transform_disp(disp)
        return left, right, disp

    def __len__(self):
        return len(self.left_files)

