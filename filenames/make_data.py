import numpy as np
import os
from glob import glob

path = '/home/autel/xzx/Data/feng/data/sceneflow/driving'
filepath = '/home/autel/xzx/unimatch/filenames'

#readpath = 'data/imgs' # 文件夹位置

#left_files = os.listdir(path) # 读取文件夹下文件名
split='frames_finalpass'

left_files = sorted(glob(path + '/' + split + '/*/*/*/left/*.png'))
#import pdb; pdb.set_trace()
# with open(os.path.join(filepath,'driving_adv.txt'),'a') as f:
#     for left_name in left_files:
#         left_img=os.path.join(left_name.split('/')[-6],left_name.split('/')[-5],left_name.split('/')[-4],left_name.split('/')[-3],left_name.split('/')[-2],left_name.split('/')[-1])
#         f.write(left_img+' ')
#         right_img = left_img.replace('left','right')
#         f.write(right_img+' ')
#         disp_img = left_img.replace(split,'disparity')[:-4]+'.pfm'
#         f.write(disp_img+'\n')

mode = 'training'
path_kitti = '/home/autel/xzx/Data/feng/data/kitti_15'
left_files_kitti = sorted(glob(path_kitti + '/' + mode + '/image_2/*_10.png'))

with open(os.path.join(filepath,'kitti15_adv.txt'),'a') as f:
    for left_name in left_files_kitti:
        left_img=os.path.join(left_name.split('/')[-3],left_name.split('/')[-2],left_name.split('/')[-1])
        f.write(left_img+' ')
        right_img = left_img.replace('image_2','image_3')
        f.write(right_img+' ')
        disp_img = left_img.replace('image_2','disp_occ_0')
        f.write(disp_img+'\n')




