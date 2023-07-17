from glob import glob
import os

mode = 'training'
path_kitti = '/home/x.zhexiao/unimatch/data/kitti_15'
left_files_kitti = sorted(glob(path_kitti + '/' + mode + '/image_2/*_11.png'))
i=0
# with open(os.path.join('kitti15_train.flist'),'a') as f:
#     #import pdb; pdb.set_trace()
#     for left_name in left_files_kitti:
#         if i%5 != 0:
#             left_img=os.path.join(left_name.split('/')[-3],left_name.split('/')[-2],left_name.split('/')[-1])
#             f.write(left_img+'\n')
#         i=i+1
#         # right_img = left_img.replace('image_2','image_3')
#         # f.write(right_img+' ')
#         # disp_img = left_img.replace('image_2','disp_occ_0')
#         # f.write(disp_img+'\n')

with open(os.path.join('kitti15_test.flist'),'a') as f:
    #import pdb; pdb.set_trace()
    for left_name in left_files_kitti:
        if i%5 == 0:
            left_img=os.path.join(left_name.split('/')[-3],left_name.split('/')[-2],left_name.split('/')[-1])
            f.write(left_img+'\n')
        i=i+1
        # right_img = left_img.replace('image_2','image_3')
        # f.write(right_img+' ')
        # disp_img = left_img.replace('image_2','disp_occ_0')
        # f.write(disp_img+'\n')