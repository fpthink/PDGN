# encoding=utf-8

import argparse
import os
import random
import numpy as np
import torch

from models.PDGN_v1 import PDGN_v1


# ----------------------------------------------

def parse_args():
    desc = "Pytorch PointGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size [default: 30]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
    parser.add_argument('--num_k',type=int, default=20,help = 'number of the knn graph point')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
    parser.add_argument('--max_epoch', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--noise_dim', type=int, default=128, help='dimensional of noise')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--debug', type=bool, default = True,  help='print log')
    parser.add_argument('--data_root', default='/test/dataset/3d_datasets/shapenetcore_partanno_segmentation_benchmark_v0/', help='data root [default: xxx]')
    #parser.add_argument('--data_root', default='/test/shapenetcore_partanno_segmentation_benchmark_v0/', help='data root [default: xxx]')
    parser.add_argument('--log_info', default='log_info.txt', help='log_info txt')
    parser.add_argument('--model_dir', help='model dir [default: None, must input]')
    parser.add_argument('--checkpoint_dir', default='checkpoint', help='Checkpoint dir [default: checkpoint]')
    parser.add_argument('--snapshot', type=int, default=20, help='how many epochs to save model')
    parser.add_argument('--choice', default=None, help='choice class')
    parser.add_argument('--network', default=None, help='which network model to be used')
    parser.add_argument('--savename',default = None,help='the generate data name')
    parser.add_argument('--pretrain_model_G', default=None, help='use the pretrain model G')
    parser.add_argument('--pretrain_model_D', default=None, help='use the pretrain model D')
    parser.add_argument('--softmax', default='True', help='softmax for bilaterl interpolation')
    parser.add_argument('--dataset', default='shapenet', help='choice dataset [shapenet, modelnet10, modelnet40]')
    return check_args(parser.parse_args())

"""
CUDA_VISIBLE_DEVICES=7 python main.py --network basecnn --choice Chair --snapshot 2 --model_dir basecnn_20190301
"""



def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def check_args(args):
    if args.model_dir is None:
        print('please create model dir')
        exit()
    if args.network is None:
        print('please select model!!!')
        exit()
    check_folder(args.checkpoint_dir)                                   # --checkpoint_dir
    check_folder(os.path.join(args.checkpoint_dir, args.model_dir))     # --chekcpoint_dir + model_dir 

    try: # --epoch
        assert args.max_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    try: # --batch_size
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def main():
    # args
    args = parse_args()
    if args is None: exit()
    args.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # create model
    print('****************network: {}****************'.format(args.network))
    if args.network == 'PDGN_v1':
        gan = PDGN_v1(args)
    else:

        print('select model error!!!')
        exit()
    # exit()

    gan.build_model()
    
    # exit()

    if args.phase == 'train' :
        # cp mainly file to corresponding model dir

        os.system('cp main.py %s' % (os.path.join(args.checkpoint_dir, args.model_dir))) # bkp of main.py
        os.system('cp models/%s.py %s' % (args.network, os.path.join(args.checkpoint_dir, args.model_dir))) # bkp of model.py

        gan.train()
        print(" [*] Training finished!")
    # exit()
    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

    if args.phase == 'cls':
        gan.extract_feature()
        print(" [*] Extract feature finished!")

if __name__ == '__main__':
    main()
