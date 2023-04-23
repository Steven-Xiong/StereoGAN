import numpy as np
import os
from glob import glob

mode = 'training'
filepath = '/home/autel/xzx/unimatch/filenames'
path_kitti = '/home/autel/xzx/Data/feng/data/kitti_15'
left_files_kitti1 = sorted(glob(path_kitti + '/' + mode + '/image_2/*_10.png'))
left_files_kitti2 = sorted(glob(path_kitti + '/' + mode + '/image_2/*_11.png'))


i = 0
for left_name1,left_name2 in zip(left_files_kitti1,left_files_kitti2):
    i=i+1
    if i % 5 == 1:
        with open(os.path.join(filepath,'kitti15_adv_flow_val.txt'),'a') as f:
            frame_id1 = os.path.basename(left_name1)
            frame_id2 = os.path.basename(left_name2)
            left_img1=os.path.join(left_name1.split('/')[-3],left_name1.split('/')[-2],left_name1.split('/')[-1])
            f.write(left_img1+' ')
            right_img1 = left_img1.replace('image_2','image_3')
            f.write(right_img1+' ')
            disp_img1 = left_img1.replace('image_2','disp_occ_0')
            f.write(disp_img1+' ')
            left_img2 = os.path.join(left_name2.split('/')[-3],left_name2.split('/')[-2],left_name2.split('/')[-1])
            f.write(left_img2 + ' ')
            flow = left_img1.replace('image_2','flow_occ')
            f.write(flow + '\n')
    else:
        with open(os.path.join(filepath,'kitti15_adv_flow_train.txt'),'a') as f:
            frame_id1 = os.path.basename(left_name1)
            frame_id2 = os.path.basename(left_name2)
            left_img1=os.path.join(left_name1.split('/')[-3],left_name1.split('/')[-2],left_name1.split('/')[-1])
            f.write(left_img1+' ')
            right_img1 = left_img1.replace('image_2','image_3')
            f.write(right_img1+' ')
            disp_img1 = left_img1.replace('image_2','disp_occ_0')
            f.write(disp_img1+' ')
            left_img2 = os.path.join(left_name2.split('/')[-3],left_name2.split('/')[-2],left_name2.split('/')[-1])
            f.write(left_img2 + ' ')
            flow = left_img1.replace('image_2','flow_occ')
            f.write(flow + '\n')



