import argparse

# def obtain_train_args():

#     # Training settings
#     parser = argparse.ArgumentParser(description='LEStereo training...')
#     LEA_group = parser.add_argument_group('LEA')
#     LEA_group.add_argument_group('--maxdisp', type=int, default=192, 
#                         help="max disp")
#     LEA_group.add_argument_group('--crop_height', type=int, default=256, 
#                         help="crop height")
#     LEA_group.add_argument_group('--crop_width', type=int, default=512, 
#                         help="crop width")
#     LEA_group.add_argument_group('--resume', type=str, default='', 
#                         help="resume from saved model")
#     LEA_group.add_argument_group('--batch_size', type=int, default=1, 
#                         help='training batch size')
#     LEA_group.add_argument_group('--testBatchSize', type=int, default=8, 
#                         help='testing batch size')
#     LEA_group.add_argument_group('--nEpochs', type=int, default=2048, 
#                         help='number of epochs to train for')
#     LEA_group.add_argument_group('--solver', default='adam',choices=['adam','sgd'],
#                         help='solver algorithms')
#     LEA_group.add_argument_group('--lr', type=float, default=0.001, 
#                         help='Learning Rate. Default=0.001')
#     LEA_group.add_argument_group('--cuda', type=int, default=1, 
#                         help='use cuda? Default=True')
#     LEA_group.add_argument_group('--threads', type=int, default=1, 
#                         help='number of threads for data loader to use')
#     LEA_group.add_argument_group('--seed', type=int, default=2019, 
#                         help='random seed to use. Default=123')
#     LEA_group.add_argument_group('--shift', type=int, default=0, 
#                         help='random shift of left image. Default=0')
#     LEA_group.add_argument_group('--save_path', type=str, default='./checkpoints/', 
#                         help="location to save models")
#     LEA_group.add_argument_group('--milestones', default=[30,50,300], metavar='N', nargs='*', 
#                         help='epochs at which learning rate is divided by 2')    
#     LEA_group.add_argument_group('--stage', type=str, default='train', choices=['search', 'train'])
#     LEA_group.add_argument_group('--dataset', type=str, default='sceneflow', 
#                         choices=['sceneflow', 'kitti15', 'kitti12', 'middlebury'], help='dataset name')

#     ######### LEStereo params ##################
#     LEA_group.add_argument_group('--fea_num_layers', type=int, default=6)
#     LEA_group.add_argument_group('--mat_num_layers', type=int, default=12)
#     LEA_group.add_argument_group('--fea_filter_multiplier', type=int, default=8)
#     LEA_group.add_argument_group('--mat_filter_multiplier', type=int, default=8)
#     LEA_group.add_argument_group('--fea_block_multiplier', type=int, default=4)
#     LEA_group.add_argument_group('--mat_block_multiplier', type=int, default=4)
#     LEA_group.add_argument_group('--fea_step', type=int, default=2)
#     LEA_group.add_argument_group('--mat_step', type=int, default=2)
#     LEA_group.add_argument_group('--net_arch_fea', default=None, type=str)
#     LEA_group.add_argument_group('--cell_arch_fea', default=None, type=str)
#     LEA_group.add_argument_group('--net_arch_mat', default=None, type=str)
#     LEA_group.add_argument_group('--cell_arch_mat', default=None, type=str)

#     args = parser.parse_args()
#     return args

def obtain_train_args():

    # Training settings
    parser = argparse.ArgumentParser(description='LEStereo training...')
    parser.add_argument('--maxdisp', type=int, default=192, 
                        help="max disp")
    parser.add_argument('--crop_height', type=int, default=256, 
                        help="crop height")
    parser.add_argument('--crop_width', type=int, default =512, 
                        help="crop width")
    parser.add_argument('--resume', type=str, default='', 
                        help="resume from saved model")
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=8, 
                        help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=2048, 
                        help='number of epochs to train for')
    parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                        help='solver algorithms')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning Rate. Default=0.001')
    parser.add_argument('--cuda', type=int, default=1, 
                        help='use cuda? Default=True')
    parser.add_argument('--threads', type=int, default=1, 
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2019, 
                        help='random seed to use. Default=123')
    parser.add_argument('--shift', type=int, default=0, 
                        help='random shift of left image. Default=0')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', 
                        help="location to save models")
    parser.add_argument('--milestones', default=[30,50,300], metavar='N', nargs='*', 
                        help='epochs at which learning rate is divided by 2')    
    parser.add_argument('--stage', type=str, default='train', choices=['search', 'train'])
    parser.add_argument('--dataset', type=str, default='sceneflow', 
                        choices=['sceneflow', 'kitti15', 'kitti12', 'middlebury'], help='dataset name')

    ######### LEStereo params ##################
    parser.add_argument('--fea_num_layers', type=int, default=6)
    parser.add_argument('--mat_num_layers', type=int, default=12)
    parser.add_argument('--fea_filter_multiplier', type=int, default=8)
    parser.add_argument('--mat_filter_multiplier', type=int, default=8)
    parser.add_argument('--fea_block_multiplier', type=int, default=4)
    parser.add_argument('--mat_block_multiplier', type=int, default=4)
    parser.add_argument('--fea_step', type=int, default=2)
    parser.add_argument('--mat_step', type=int, default=2)
    parser.add_argument('--net_arch_fea', default='pretrained_LEA/sceneflow/experiment/feature_network_path.npy', type=str)
    parser.add_argument('--cell_arch_fea', default='pretrained_LEA/sceneflow/experiment/feature_genotype.npy', type=str)
    parser.add_argument('--net_arch_mat', default='pretrained_LEA/sceneflow/experiment/matching_network_path.npy', type=str)
    parser.add_argument('--cell_arch_mat', default='pretrained_LEA/sceneflow/experiment/matching_genotype.npy', type=str)

    args = parser.parse_args()
    return args