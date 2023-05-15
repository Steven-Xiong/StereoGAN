import glob
import random
import os
import png
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from utils.util import read_all_lines, pfm_imread,readflow_driving,readFlowKITTI, read_vkitti2_flow, read_vkitti2_disp
from augmentation import FlowAugmentor, SparseFlowAugmentor

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

# crop to (256, 512), both left and right images, conditioned on disp
class ImageDataset(Dataset):
    def __init__(self, rootA='data/sceneflow/driving', rootB='data/kitti_15', 
                 rootC = '' , rootD ='', height=320, width=512, transforms_=None,left_right_consistency=1):
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
        # self.leftA_files, self.rightA_files, self.dispA_files = self.load_path('filenames/driving_adv.txt')
        # self.leftB_files, self.rightB_files, self.dispB_files = self.load_path('filenames/kitti15_adv.txt')
        self.leftA_files, self.rightA_files, self.dispA_files,self.leftA_forward,self.flowA = self.load_flow_path('filenames/driving_adv_flow_debug.txt')
        self.leftB_files, self.rightB_files, self.dispB_files, self.leftB_forward,self.flowB = self.load_flow_path('filenames/kitti15_adv_flow_train_debug.txt')
        
        # self.augmentorA = FlowAugmentor({'crop_size': [256, 512], 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True})
        # self.augmentorB = SparseFlowAugmentor({'crop_size': [256, 512], 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False})
        

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

    def load_flow_path(self, list_filename):
        #import pdb; pdb.set_trace()
        lines = read_all_lines(list_filename)
    
        splits = [line.split() for line in lines]
        #print(len(splits))
        # for x in splits:
        #     print(x)
        #print([x[0] for x in splits])
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        left_forward_images = [x[3] for x in splits]
        flow_images = [x[4] for x in splits]
        return left_images, right_images, disp_images, left_forward_images,flow_images

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
    def load_flowA(self, filename):
        # data, scale = pfm_imread(filename)
        # data = np.ascontiguousarray(data, dtype=np.float32)
        # if len(data.shape) == 2:
        #     return data
        # else:
        #     return data[:, :, :-1]
        # #return data
        data = readflow_driving(filename)
        return data
        

    def load_flowB(self, filename):
        # data = Image.open(filename)
        # data = np.array(data, dtype=np.float32) / 256.
        # return data
        data,valid = readFlowKITTI(filename)
        return data,valid
        
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
        
        leftA_forward = self.load_image(os.path.join(self.rootA, self.leftA_forward[index]))
        leftB_forward = self.load_image(os.path.join(self.rootB, self.leftB_forward[index2]))
        flowA = self.load_flowA(os.path.join(self.rootA,self.flowA[index]))
        flowB,validB = self.load_flowB(os.path.join(self.rootB,self.flowB[index2]))
        # print('flowA的shape:',flowA.shape)
        # validA = (abs(flowA[0]) < 1000) & (abs(flowA[1]) < 1000)
        #validA = validA.float()

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
        leftA_forward = leftA_forward.crop((x1, y1, x1+crop_w, y1+crop_h))
        leftB_forward = leftB_forward.crop((x2, y2, x2+crop_w, y2+crop_h))
        #print('flowB.shape',flowB.shape)
        flowA = flowA[y1:y1+crop_h, x1:x1+crop_w]
        flowB = flowB[y2:y2+crop_h, x2:x2+crop_w]
        #validA = validA[y1:y1+crop_h, x1:x1+crop_w]
        validB = validB[y2:y2+crop_h, x2:x2+crop_w]

        if self.left_right_consistency:
            error_mapB = error_mapB[y2:y2+crop_h, x2:x2+crop_w]
        else:
            error_mapB = []

        # # augmentation:
        # leftA, leftA_forward, flowA = self.augmentorA(leftA, leftA_forward, flowA)
        # leftB, leftB_forward, flowB, validB = self.augmentorB(leftB, leftB_forward, flowB, validB) 

        # transform
        leftA = self.transform(leftA)
        rightA = self.transform(rightA)
        leftB = self.transform(leftB)
        rightB = self.transform(rightB)

        leftA_forward = self.transform(leftA_forward)
        leftB_forward = self.transform(leftB_forward)

        return {"leftA": leftA, "rightA": rightA, "dispA": dispA,'leftA_forward':leftA_forward,'flowA': flowA, \
        "leftB": leftB, "rightB": rightB, "dispB": dispB, "error_mapB": error_mapB,'leftB_forward':leftB_forward, 'flowB': flowB,\
        "validA": [] , "validB": validB}

    def __len__(self):
        return len(self.leftA_files)

# for validation
class ValJointImageDataset(Dataset):
    def __init__(self, root='data/kitti_15', transforms_=None, input_shape=(3, 384, 1280)):
        f = open('./filenames/kitti15_adv_flow_val_debug.txt', 'r')
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
        self.left_forward_files = []
        self.flow_files = []
        for line in f:
            line = line.strip()
            a, b, c ,d ,e= line.split()
            self.left_files.append(os.path.join(root, a))
            self.right_files.append(os.path.join(root, b))
            self.disp_files.append(os.path.join(root, c))
            self.left_forward_files.append(os.path.join(root,d))
            self.flow_files.append(os.path.join(root,e))

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data
        
    def load_flowB(self, filename):
        # data = Image.open(filename)
        # data = np.array(data, dtype=np.float32) / 256.
        # return data
        data,valid = readFlowKITTI(filename)
        return data,valid

    def __getitem__(self, index):
        left = self.load_image(self.left_files[index])
        shape = np.array(left).shape[:2]
        right = self.load_image(self.right_files[index])
        disp = self.load_disp(self.disp_files[index])
        #disp = Image.fromarray(disp)
        # add flow
        left_forward = self.load_image(self.left_forward_files[index])
        flow,valid = self.load_flowB(self.flow_files[index])  #B 返回flow和对应valid值
        
        top_pad = 384 - shape[0]
        right_pad = 1280 - shape[1]    #1280
        assert top_pad > 0 and right_pad > 0
        # pad disparity gt
        #print('disp.shape',disp.shape)
        
        if disp is not None:
            assert len(disp.shape) == 2
            disp = np.lib.pad(disp, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        
        
        left = self.transform(left).numpy()
        right = self.transform(right).numpy()
        left_forward = self.transform(left_forward).numpy()
        
        #print('left.shape',left.shape)
        #print(len(flow))
        #print('flow.shape',flow.shape)
        #print('flow[1].shape',flow[1].shape)
        flow = np.transpose(flow,(2,0,1))
        #print('flow.shape',flow.shape)
        left = np.lib.pad(left, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        right = np.lib.pad(right, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        left_forward = np.lib.pad(left_forward, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        flow = np.lib.pad(flow, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        valid = np.lib.pad(valid, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        
        #disp = self.transform_disp(disp)
        #print(left.shape,right.shape, disp.shape, left_forward.shape, flow.shape, valid.shape)
        return left, right, disp, left_forward, flow, valid

    def __len__(self):
        return len(self.left_files)

# For VKITTI2 and KITTI15 crop to (256, 512), both left and right images, conditioned on disp
class ImageDataset2(Dataset):
    def __init__(self, rootA='data/VKITTI2', rootB='data/kitti_15', 
                 rootC = '' , rootD ='', height=320, width=512, transforms_=None,left_right_consistency=1):
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
        # self.leftA_files, self.rightA_files, self.dispA_files = self.load_path('filenames/driving_adv.txt')
        # self.leftB_files, self.rightB_files, self.dispB_files = self.load_path('filenames/kitti15_adv.txt')
        self.leftA_files, self.rightA_files, self.dispA_files,self.leftA_forward,self.flowA = self.load_flow_path('filenames/VKITTI2_adv_flow.txt')
        self.leftB_files, self.rightB_files, self.dispB_files, self.leftB_forward,self.flowB = self.load_flow_path('filenames/kitti15_adv_flow_train.txt')

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

    def load_flow_path(self, list_filename):
        #import pdb; pdb.set_trace()
        lines = read_all_lines(list_filename)
    
        splits = [line.split() for line in lines]
    
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        left_forward_images = [x[3] for x in splits]
        flow_images = [x[4] for x in splits]
        return left_images, right_images, disp_images, left_forward_images,flow_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data
    # change to vkitti2 format
    def load_dispA(self, filename):
        disp = read_vkitti2_disp(filename)
        return disp
    
    def load_flowA(self, filename):
        # data, scale = pfm_imread(filename)
        # data = np.ascontiguousarray(data, dtype=np.float32)
        # if len(data.shape) == 2:
        #     return data
        # else:
        #     return data[:, :, :-1]
        # #return data
        data,valid = read_vkitti2_flow(filename)
        return data, valid
        

    def load_flowB(self, filename):
        # data = Image.open(filename)
        # data = np.array(data, dtype=np.float32) / 256.
        # return data
        data,valid = readFlowKITTI(filename)
        return data,valid
        
    def __getitem__(self, index):
        index2 = random.randint(0, len(self.leftB_files) - 1)
        leftA = self.load_image(os.path.join(self.rootA, self.leftA_files[index]))
        rightA = self.load_image(os.path.join(self.rootA, self.rightA_files[index]))
        dispA = self.load_dispA(os.path.join(self.rootA, self.dispA_files[index])) # 改为vkitti2
        leftB = self.load_image(os.path.join(self.rootB, self.leftB_files[index2]))
        rightB = self.load_image(os.path.join(self.rootB, self.rightB_files[index2]))
        if self.left_right_consistency:
            dispB = self.load_disp(os.path.join(self.rootB_pseudo_label, self.dispB_files[index2].split('/')[-1]))
            error_mapB = self.load_dispA(os.path.join(self.rootB_error_map, self.dispB_files[index2].split('/')[-1].replace('.png','.pfm')))
        else:
            dispB = self.load_disp(os.path.join(self.rootB, self.dispB_files[index2]))
            error_mapB = []
        
        leftA_forward = self.load_image(os.path.join(self.rootA, self.leftA_forward[index]))
        leftB_forward = self.load_image(os.path.join(self.rootB, self.leftB_forward[index2]))
        flowA, validA = self.load_flowA(os.path.join(self.rootA,self.flowA[index]))
        flowB, validB = self.load_flowB(os.path.join(self.rootB,self.flowB[index2]))
        # print('flowA的shape:',flowA.shape)
        # validA = (abs(flowA[0]) < 1000) & (abs(flowA[1]) < 1000)
        #validA = validA.float()

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
        leftA_forward = leftA_forward.crop((x1, y1, x1+crop_w, y1+crop_h))
        leftB_forward = leftB_forward.crop((x2, y2, x2+crop_w, y2+crop_h))
        #print('flowB.shape',flowB.shape)
        flowA = flowA[y1:y1+crop_h, x1:x1+crop_w]
        flowB = flowB[y2:y2+crop_h, x2:x2+crop_w]
        validA = validA[y1:y1+crop_h, x1:x1+crop_w]
        validB = validB[y2:y2+crop_h, x2:x2+crop_w]

        if self.left_right_consistency:
            error_mapB = error_mapB[y2:y2+crop_h, x2:x2+crop_w]
        else:
            error_mapB = []
        # transform
        leftA = self.transform(leftA)
        rightA = self.transform(rightA)
        leftB = self.transform(leftB)
        rightB = self.transform(rightB)

        leftA_forward = self.transform(leftA_forward)
        leftB_forward = self.transform(leftB_forward)

        return {"leftA": leftA, "rightA": rightA, "dispA": dispA,'leftA_forward':leftA_forward,'flowA': flowA, \
        "leftB": leftB, "rightB": rightB, "dispB": dispB, "error_mapB": error_mapB,'leftB_forward':leftB_forward, 'flowB': flowB,\
        "validA": validA , "validB": validB}

    def __len__(self):
        return len(self.leftA_files)
    
# for validation
class ValJointImageDataset2(Dataset):
    def __init__(self, root='data/kitti_15', transforms_=None, input_shape=(3, 384, 1280)):
        f = open('./filenames/kitti15_adv_flow_val.txt', 'r')
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
        self.left_forward_files = []
        self.flow_files = []
        for line in f:
            line = line.strip()
            a, b, c ,d ,e= line.split()
            self.left_files.append(os.path.join(root, a))
            self.right_files.append(os.path.join(root, b))
            self.disp_files.append(os.path.join(root, c))
            self.left_forward_files.append(os.path.join(root,d))
            self.flow_files.append(os.path.join(root,e))

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data
        
    def load_flowB(self, filename):
        # data = Image.open(filename)
        # data = np.array(data, dtype=np.float32) / 256.
        # return data
        data,valid = readFlowKITTI(filename)
        return data,valid

    def __getitem__(self, index):
        left = self.load_image(self.left_files[index])
        shape = np.array(left).shape[:2]
        right = self.load_image(self.right_files[index])
        disp = self.load_disp(self.disp_files[index])
        #disp = Image.fromarray(disp)
        # add flow
        left_forward = self.load_image(self.left_forward_files[index])
        flow,valid = self.load_flowB(self.flow_files[index])  #B 返回flow和对应valid值
        #import pdb; pdb.set_trace()
        
        # pad disparity gt
        #print('disp.shape',disp.shape)
        
        left = self.transform(left).numpy()
        right = self.transform(right).numpy()
        left_forward = self.transform(left_forward).numpy()
        
        # left = left.numpy()
        # right = right.numpy()
        # left_forward = left_forward.numpy()

        # #print('left.shape',left.shape)
        # #print(len(flow))
        # #print('flow.shape',flow.shape)
        # #print('flow[1].shape',flow[1].shape)
        # flow = np.transpose(flow,(2,0,1))
        # #print('flow.shape',flow.shape)
        # left = np.lib.pad(left, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        # right = np.lib.pad(right, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        # left_forward = np.lib.pad(left_forward, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        # flow = np.lib.pad(flow, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        # valid = np.lib.pad(valid, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        return left, right, disp, left_forward, flow, valid

    def __len__(self):
        return len(self.left_files)