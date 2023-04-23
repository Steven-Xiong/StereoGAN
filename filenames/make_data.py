import numpy as np
import os
from glob import glob
'''
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
'''
#DRIVINGS
# path = '/home/autel/xzx/Data/feng/data/sceneflow/driving'
# filepath = '/home/autel/xzx/unimatch/filenames'

# #readpath = 'data/imgs' # 文件夹位置

# #left_files = os.listdir(path) # 读取文件夹下文件名
# split='frames_finalpass'

# flow_state  = 'into_future'
# left_files = sorted(glob(path + '/' + split + '/*/*/*/left/*.png'))
# #import pdb; pdb.set_trace()
# with open(os.path.join(filepath,'driving_adv_flow.txt'),'a') as f:
#     for left_name in left_files:
#         left_img=os.path.join(left_name.split('/')[-6],left_name.split('/')[-5],left_name.split('/')[-4],left_name.split('/')[-3],left_name.split('/')[-2],left_name.split('/')[-1])
#         f.write(left_img+' ')
#         right_img = left_img.replace('left','right')
#         f.write(right_img+' ')
#         disp_img = left_img.replace(split,'disparity')[:-4]+'.pfm'
#         f.write(disp_img+' ')
#         #import pdb; pdb.set_trace()
#         index1 = int(left_img.replace(split,'optical_flow').split('/')[-1].replace('.png',''))
#         index2 = index1+1
#         index1= str(index1).zfill(4)
#         index2 = str(index2).zfill(4)
#         left_img_forward = os.path.join(left_name.split('/')[-6],left_name.split('/')[-5],left_name.split('/')[-4],left_name.split('/')[-3],left_name.split('/')[-2],left_name.split('/')[-1].replace(index1,index2))
#         f.write(left_img_forward+' ')
#         flow_img = os.path.join('optical_flow', left_name.split('/')[-5],left_name.split('/')[-4],left_name.split('/')[-3],flow_state, \
#             left_name.split('/')[-2],'OpticalFlowIntoFuture_'+left_img.replace(split,'optical_flow').split('/')[-1].replace('.png','')+'_L.pfm')
#         f.write(flow_img+'\n')

mode = 'training'
filepath = '/home/autel/xzx/unimatch/filenames'
path_kitti = '/home/autel/xzx/Data/feng/data/kitti_15'
left_files_kitti1 = sorted(glob(path_kitti + '/' + mode + '/image_2/*_10.png'))
left_files_kitti2 = sorted(glob(path_kitti + '/' + mode + '/image_2/*_11.png'))
#也可以直接replace
with open(os.path.join(filepath,'kitti15_adv_flow.txt'),'a') as f:
    for left_name1,left_name2 in zip(left_files_kitti1,left_files_kitti2):
        #import pdb; pdb.set_trace()
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


