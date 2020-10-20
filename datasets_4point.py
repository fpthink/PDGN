# encoding=utf-8

import torch.utils.data as data
import os
# import errno
import torch
# import json
# import codecs
import numpy as np
# import sys
# import torchvision.transforms as transforms
# import argparse
import os.path
import json
import numpy as np
import sys
import re
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider

class PartDataset(data.Dataset):
    def __init__(self, root, batch_size, npoints1 = 256,npoints2 = 512,npoints3 = 1024,npoints4 = 2048, classification=False, class_choice=None, train=True):
        self.npoints1 = npoints1
        self.npoints2 = npoints2
        self.npoints3 = npoints3
        self.npoints4 = npoints4
        self.root = root
        self.batch_size = batch_size
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        self.classification = classification

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]

            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))


        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath) // self.batch_size):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)
        if len(seg)>self.npoints4:
            choice1 = np.random.choice(len(seg), self.npoints1, replace=False)
            choice2 = np.random.choice(len(seg), self.npoints2, replace=False)
            choice3 = np.random.choice(len(seg), self.npoints3, replace=False)
            choice4 = np.random.choice(len(seg), self.npoints4, replace=False)
        else:
            choice1 = np.random.choice(len(seg), self.npoints1, replace=True)
            choice2 = np.random.choice(len(seg), self.npoints2, replace=True)
            choice3 = np.random.choice(len(seg), self.npoints3, replace=True)
            choice4 = np.random.choice(len(seg), self.npoints4, replace=True)
        #resample
        point_set1 = point_set[choice1, :]
        point_set2 = point_set[choice2, :]
        point_set3 = point_set[choice3, :]
        point_set4 = point_set[choice4, :]
        seg = seg[choice3]
        point_set1 = torch.from_numpy(point_set1)
        point_set2 = torch.from_numpy(point_set2)
        point_set3 = torch.from_numpy(point_set3)
        point_set4 = torch.from_numpy(point_set4)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set1,point_set2,point_set3,point_set4,cls
        else:
            return point_set1,point_set2,point_set3,point_set4,seg

    def __len__(self):
        return len(self.datapath)

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetDataset():
    def __init__(self, root, batch_size = 50, npoints1 = 256,npoints2 = 512,npoints3 = 1024,npoints4 = 2048, split='train', normalize=True, normal_channel=False, modelnet10=False, cache_size=15000, shuffle=None,class_choice=None):
        self.root = root+'modelnet40_normal_resampled/'
        self.batch_size = batch_size
        self.npoints1 = npoints1
        self.npoints2 = npoints2
        self.npoints3 = npoints3
        self.npoints4 = npoints4
        self.normalize = normalize
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.normal_channel = normal_channel
        
        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert(split=='train' or split=='test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (point_set, cls) tuple

        if shuffle is None:
            if split == 'train': self.shuffle = True
            else: self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset()

    def _augment_batch_data(self, batch_data):
        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
    
        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        return provider.shuffle_points(rotated_data)


    def _get_item(self, index): 
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)
            # Take the first npoints
            point_set1 = point_set[0:self.npoints1,:]
            point_set2 = point_set[0:self.npoints2,:]
            point_set3 = point_set[0:self.npoints3,:]
            point_set4 = point_set[0:self.npoints4,:]
            if self.normalize:
                point_set1[:,0:3] = pc_normalize(point_set1[:,0:3])
                point_set2[:,0:3] = pc_normalize(point_set2[:,0:3])
                point_set3[:,0:3] = pc_normalize(point_set3[:,0:3])
                point_set4[:,0:3] = pc_normalize(point_set4[:,0:3])
            if not self.normal_channel:
                point_set1 = point_set1[:,0:3]
                point_set2 = point_set2[:,0:3]
                point_set3 = point_set3[:,0:3]
                point_set4 = point_set4[:,0:3]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
        return point_set1,point_set2,point_set3,point_set4,cls
        
    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3

    def reset(self):
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath)+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.npoints, self.num_channel()))
        batch_label = np.zeros((bsize), dtype=np.int32)
        for i in range(bsize):
            ps,cls = self._get_item(self.idxs[i+start_idx])
            batch_data[i] = ps
            batch_label[i] = cls
        self.batch_idx += 1
        if augment: batch_data = self._augment_batch_data(batch_data)
        return batch_data, batch_label
if __name__ == '__main__':
    print('test')
    d = PartDataset(root = '/test/shapenetcore_partanno_segmentation_benchmark_v0/',batch_size = 50,class_choice = 'Chair')
    # d = ModelNetDataset(root = '/data3/3D_Datasets/modelnet40_normal_resampled',batch_size = 50,modelnet10=True)
    real_data1,real_data2,real_data3,real_data4 = [],[],[],[]
    i = 0
    for idx, data in enumerate(d, 5):
        point1,point2,point3,point4,_ = data
        point1 = point1.numpy()
        point2 = point2.numpy()
        point3 = point3.numpy()
        point4 = point4.numpy()
        real_data1.append(point1)
        real_data2.append(point2)
        real_data3.append(point3)
        real_data4.append(point4)
        i+=1
        print(np.shape(real_data1))
        if i == 60 :
            break
    # np.save('./real_data1_guitar',real_data1)
    # np.save('./real_data2_guitar',real_data2)
    # np.save('./real_data3_guitar',real_data3)
    np.save('./real_data1_chair',real_data1)


#     d = PartDataset(root = 'shapenetcor_partanno_segmentation_benchmark_v0', classification = True)
#     print(len(d))
#     ps, cls = d[0]
#     print(ps.size(), ps.type(), cls.size(),cls.type())
