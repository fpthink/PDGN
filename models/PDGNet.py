# encoding=utf-8

import numpy as np
import math
import sys
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

# add for shape-preserving Loss
from lib.pointops.functions import pointops

from datasets_4point import ShapeNetCore,ModelNetDataset
from collections import namedtuple
from utils import chamfer_loss
cudnn.benchnark=True

class PDGNet(object):
    def __init__(self, args):
        self.model_name = args.network
        self.workers = args.workers
        self.checkpoint_dir = args.checkpoint_dir
        self.model_dir = args.model_dir
        self.data_root = args.data_root
        self.pretrain_model_G = args.pretrain_model_G
        self.pretrain_model_D = args.pretrain_model_D
        self.normalize = args.normalize
        self.seed = args.seed
        # softmax for bilaterl interpolation
        if args.softmax == 'True':
            self.softmax = True
            print('use softmax')
        else:
            self.softmax = False
            print('do not use softmax')
  
        self.epoch = args.max_epoch             # 300
        self.batch_size = args.batch_size       # 50
        self.noise_dim = args.noise_dim         # 128
        self.learning_rate = args.learning_rate # 0.0001
        self.num_point =   args.num_point      # 2048
        self.num_k =  args.num_k               # 20
        self.choice = args.choice               # which class
        self.snapshot = args.snapshot           # 20 epochs / one
        self.savename = args.savename
        self.save_dir = args.save_dir
        self.device = args.device
        if self.choice is None:
            self.category = 'full'
        else:
            self.category = self.choice

        self.chamfer_loss = chamfer_loss.ChamferLoss()

        if args.dataset == 'shapenet15k':
            print('-------------use dataset shapenet15k-------------')
            self.dataset = ShapeNetCore(path=self.data_root, cates_list=self.choice,split='train',scale_mode='shape_unit')
           
        elif args.dataset == 'modelnet10':
            print('-------------use dataset modelnet10-------------')
            self.dataset = ModelNetDataset(root=self.data_root, batch_size=self.batch_size, npoints=self.num_point, 
                                           split='train', normalize=True, normal_channel=False, modelnet10=True,class_choice=self.choice)
            
        elif args.dataset == 'modelnet40':
            print('-------------use dataset modelnet40-------------')
            self.dataset = ModelNetDataset(root=self.data_root, batch_size=self.batch_size, npoints=self.num_point,
                                           split='train', normalize=True, normal_channel=False, modelnet10=False,class_choice=self.choice)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=int(self.workers))        
        self.num_batches = len(self.dataset) // self.batch_size

        if args.phase == 'train':
            print('training...')
            self.log_info = args.log_info
            self.LOG_FOUT = open(os.path.join(self.checkpoint_dir, self.model_dir, self.log_info), 'w')
            self.LOG_FOUT.write(str(args)+'\n')
        elif args.phase == 'test':
            print('testing...')
        elif args.phase == 'cls':
            print('extract feature')
        
        cudnn.benchmark = True  # cudnn

    def build_model(self):
        """ Models """
        self.generator = PointGenerator(self.num_point, self.num_k, self.softmax)
        self.discriminator1 = PointDiscriminator_1()
        self.discriminator2 = PointDiscriminator_2()
        self.discriminator3 = PointDiscriminator_3()
        self.discriminator4 = PointDiscriminator_4()

        self.generator = nn.DataParallel(self.generator)
        self.discriminator1 = nn.DataParallel(self.discriminator1)
        self.discriminator2 = nn.DataParallel(self.discriminator2)
        self.discriminator3 = nn.DataParallel(self.discriminator3)
        self.discriminator4 = nn.DataParallel(self.discriminator4)
        
        self.generator.cuda()
        self.discriminator1.cuda()
        self.discriminator2.cuda()
        self.discriminator3.cuda()
        self.discriminator4.cuda()

        """ Loss Function """
        #self.group = pointops.Gen_QueryAndGroupXYZ(radius=0.1, nsample=10, use_xyz=False)
        self.group = pointops.Gen_QueryAndGroupXYZ(radius=None, nsample=20, use_xyz=False)
        self.loss_fn = nn.MSELoss()
        self.shape_loss_fn = nn.MSELoss()

        """ Training """
        
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizerD1 = optim.Adam(self.discriminator1.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizerD2 = optim.Adam(self.discriminator2.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizerD3 = optim.Adam(self.discriminator3.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizerD4 = optim.Adam(self.discriminator4.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
    
    def compute_mean_covariance(self, points):
        bs, ch, nump = points.size()
        # ----------------------------------------------------------------
        mu = points.mean(dim=-1, keepdim=True)  # Bx3xN -> Bx3x1
        # ----------------------------------------------------------------
        tmp = points - mu.repeat(1, 1, nump)    # Bx3xN - Bx3xN -> Bx3xN
        tmp_transpose = tmp.transpose(1, 2)     # Bx3xN -> BxNx3
        covariance = torch.bmm(tmp, tmp_transpose)
        covariance = covariance / nump
        return mu, covariance   # Bx3x1 Bx3x3

    def get_local_pair(self, pt1, pt2):
        pt1_batch,pt1_N,pt1_M = pt1.size()
        pt2_batch,pt2_N,pt2_M = pt2.size()
        # pt1: Bx3xM    pt2: Bx3XN      (N > M)
        #print('pt1: {}      pt2: {}'.format(pt1.size(), pt2.size()))
        new_xyz = pt1.transpose(1, 2).contiguous()      # Bx3xM -> BxMx3
        pt1_trans = pt1.transpose(1, 2).contiguous()    # Bx3xM -> BxMx3
        pt2_trans = pt2.transpose(1, 2).contiguous()    # Bx3xN -> BxNx3
        
        g_xyz1 = self.group(pt1_trans, new_xyz)     # Bx3xMxK  
        g_xyz2 = self.group(pt2_trans, new_xyz)     # Bx3xMxK
        
        g_xyz1 = g_xyz1.transpose(1, 2).contiguous().view(-1, 3, 20)    # Bx3xMxK -> BxMx3xK -> (BM)x3xK
        g_xyz2 = g_xyz2.transpose(1, 2).contiguous().view(-1, 3, 20)    # Bx3xMxK -> BxMx3xK -> (BM)x3xK
        # print('====================== FPS ========================')
        mu1, var1 = self.compute_mean_covariance(g_xyz1) 
        mu2, var2 = self.compute_mean_covariance(g_xyz2) 
        mu1 = mu1.view(pt1_batch,-1,3)
        mu2 = mu2.view(pt2_batch,-1,3)
        var1 = var1.view(pt1_batch,-1,9)
        var2 = var2.view(pt2_batch,-1,9)
        like_mu12 = self.chamfer_loss(mu1,mu2) / float(pt1_M)
        like_var12 = self.chamfer_loss(var1,var2) / float(pt1_M)
       
        return like_mu12, like_var12

    def train(self):
        # restore check-point if it exits
        could_load, save_epoch = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 1
            print(" [!] start epoch: {}".format(start_epoch))

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch+1):
            for idx, data in enumerate(self.dataloader, 0):
                if idx+1 > self.num_batches: continue
                # exit()
                # ----------------train D-----------------------
                points1, points2, points3, points4, _ = data
                points1 = Variable(points1)
                points2 = Variable(points2)
                points3 = Variable(points3)
                points4 = Variable(points4)
                target = Variable(torch.from_numpy(np.ones(self.batch_size,).astype(np.int64))).cuda().float().reshape(self.batch_size, 1)
                
                sim_noise = Variable(torch.Tensor(np.random.normal(0, 0.2, (self.batch_size, self.noise_dim)))).cuda()
                fake1, fake2, fake3, fake4 = self.generator(sim_noise)
                fake_target = Variable(torch.from_numpy(np.zeros(self.batch_size,).astype(np.int64))).cuda().float().reshape(self.batch_size, 1)
                
                # ------------------D1---------------
                self.optimizerD1.zero_grad()
                # self.discriminator1.zero_grad()
                points1 = points1.transpose(2, 1).cuda()
                pred1 = self.discriminator1(points1)
                pred1_fake = self.discriminator1(fake1.detach())
                loss1_1 = self.loss_fn(pred1, target)
                loss2_1 = self.loss_fn(pred1_fake, fake_target)
                lossD1 = (loss1_1 + loss2_1) / 2.0
                lossD1.backward()
                self.optimizerD1.step()
                
                # ------------------D2---------------
                self.optimizerD2.zero_grad()
                points2 = points2.transpose(2, 1).cuda()
                pred2 = self.discriminator2(points2)
                pred2_fake = self.discriminator2(fake2.detach())
                loss1_2 = self.loss_fn(pred2, target)
                loss2_2 = self.loss_fn(pred2_fake, fake_target)
                lossD2 = (loss1_2 + loss2_2) / 2.0
                lossD2.backward()
                self.optimizerD2.step()

                # ------------------D3---------------
                self.optimizerD3.zero_grad()
                points3 = points3.transpose(2, 1).cuda()
                pred3 = self.discriminator3(points3)
                pred3_fake = self.discriminator3(fake3.detach())
                loss1_3 = self.loss_fn(pred3, target)
                loss2_3 = self.loss_fn(pred3_fake, fake_target)
                lossD3 = (loss1_3 + loss2_3) / 2.0
                lossD3.backward()
                self.optimizerD3.step()
                
                # ------------------D4---------------
                self.optimizerD4.zero_grad()
                points4 = points4.transpose(2, 1).cuda()
                pred4 = self.discriminator4(points4)
                pred4_fake = self.discriminator4(fake4.detach())
                loss1_4 = self.loss_fn(pred4, target)
                loss2_4 = self.loss_fn(pred4_fake, fake_target)
                lossD4 = (loss1_4 + loss2_4) / 2.0
                lossD4.backward()
                self.optimizerD4.step()
                
                # -----------------------------------train G-----------------------------------
                self.optimizerG.zero_grad()
                sim_noise = Variable(torch.Tensor(np.random.normal(0, 0.2, (self.batch_size, self.noise_dim)))).cuda()
                points1_gen, points2_gen, points3_gen, points4_gen = self.generator(sim_noise)
                # p1=Bx3x256        p2=Bx3x512      p3=Bx3x1024     p4=Bx3x2048 
                #print('points1_gen: {}'.format(points1_gen.size()))
                
                like_mu12, like_cov12 = self.get_local_pair(points1_gen, points2_gen)
                like_mu13, like_cov13 = self.get_local_pair(points1_gen, points3_gen)
                like_mu14, like_cov14 = self.get_local_pair(points1_gen, points4_gen)
                like_mu23, like_cov23 = self.get_local_pair(points2_gen, points3_gen)
                like_mu24, like_cov24 = self.get_local_pair(points2_gen, points4_gen)
                like_mu34, like_cov34 = self.get_local_pair(points3_gen, points4_gen)

                pred_g1 = self.discriminator1(points1_gen)
                pred_g2 = self.discriminator2(points2_gen)
                pred_g3 = self.discriminator3(points3_gen)
                pred_g4 = self.discriminator4(points4_gen)
                target_g = Variable(torch.from_numpy(np.ones(self.batch_size, ).astype(np.int64))).cuda().float().reshape(self.batch_size, 1)
                
                g_loss_1 = self.loss_fn(pred_g1, target_g)
                g_loss_2 = self.loss_fn(pred_g2, target_g)
                g_loss_3 = self.loss_fn(pred_g3, target_g)
                g_loss_4 = self.loss_fn(pred_g4, target_g)

                w = 1.0
                similar_loss = w * 1.0 * (like_mu12 + like_mu13 + like_mu14 + like_mu23 + like_mu24 + like_mu34) + \
                               w * 5.0 * (like_cov12 + like_cov13 + like_cov14 + like_cov23 + like_cov24 + like_cov34)
                lossG = (1.2*g_loss_1 + 1.2*g_loss_2 + 1.2*g_loss_3 + g_loss_4) + 0.5*similar_loss

                lossG.backward()
                self.optimizerG.step()
                
                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %2dm %2ds d_loss1: %.8f d_loss2: %.8f d_loss3: %.8f d_loss4: %.8f, g_loss: %.8f, similar_loss: %.8f" \
                      % (epoch, idx+1, self.num_batches, (time.time()-start_time)/60,(time.time()-start_time)%60, 
                         lossD1.item(), lossD2.item(), lossD3.item(), lossD4.item(), lossG.item(), similar_loss.item()))
                self.log_string("Epoch: [%2d] [%4d/%4d] time: %2dm %2ds d_loss1: %.8f d_loss2: %.8f d_loss3: %.8f d_loss4: %.8f, g_loss: %.8f, similar_loss: %.8f" \
                      % (epoch, idx+1, self.num_batches, (time.time()-start_time)/60,(time.time()-start_time)%60, 
                         lossD1.item(), lossD2.item(), lossD3.item(), lossD4.item(), lossG.item(), similar_loss.item()))

            if epoch % self.snapshot == 0:
                self.save(self.checkpoint_dir, epoch)
        self.save(self.checkpoint_dir, self.epoch)
        self.LOG_FOUT.close()

    def test(self):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        save_dir = os.path.join(self.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(self.choice), int(time.time())) )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger = get_logger('test', save_dir)
        seed_all(self.seed)

        # Datasets and loaders
        logger.info('Loading datasets...')
        test_dset = ShapeNetCore(path=self.data_root,cates_list=self.choice,split='test',scale_mode=self.normalize,
        )
        test_loader = DataLoader(test_dset, batch_size=self.batch_size, num_workers=0)

        # Model
        logger.info('Loading model...')
       
        # Reference Point Clouds
        ref_pcs = []
        for i, data in enumerate(test_dset):
            _,_,_,p4,_ = data
            ref_pcs.append(p4.unsqueeze(0))
        ref_pcs = torch.cat(ref_pcs, dim=0)

        # Generate Point Clouds
        gen_pcs = []
        for i in tqdm(range(0, math.ceil(len(test_dset) / self.batch_size)), 'Generate'):
            with torch.no_grad():
                z = torch.randn([self.batch_size, 128]).cuda()
                _,_,_,x = self.generator(z)
                x = x.transpose(2, 1)
                gen_pcs.append(x.detach().cpu())
        gen_pcs = torch.cat(gen_pcs, dim=0)[:len(test_dset)]
        np.save(os.path.join(save_dir, 'nonormal_out.npy'), gen_pcs.numpy())
        if self.normalize is not None:
            gen_pcs = self.normalize_point_clouds(gen_pcs, mode=self.normalize, logger=logger)

        # Save
        logger.info('Saving point clouds...')
        np.save(os.path.join(save_dir, 'out.npy'), gen_pcs.numpy())

        # Compute metrics
        with torch.no_grad():
            results = compute_all_metrics(gen_pcs.to(self.device), ref_pcs.to(self.device), self.batch_size)
            results = {k:v.item() for k, v in results.items()}
            jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
            results['jsd'] = jsd

        for k, v in results.items():
            logger.info('%s: %.12f' % (k, v))

    def log_string(self, out_str):
        self.LOG_FOUT.write(out_str+'\n')
        self.LOG_FOUT.flush()
        # print(out_str)

    def load(self, checkpoint_dir):
        if self.pretrain_model_G is None  and self.pretrain_model_D is None:
            print('################ new training ################')
            return False, 1

        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        
        # ----------------- load G -------------------
        if not self.pretrain_model_G is None:
            resume_file_G = os.path.join(checkpoint_dir, self.pretrain_model_G)
            flag_G = os.path.isfile(resume_file_G), 
            if flag_G == False:
                print('G--> Error: no checkpoint directory found!')
                exit()
            else:
                print('resume_file_G------>: {}'.format(resume_file_G))
                checkpoint = torch.load(resume_file_G)
                self.generator.load_state_dict(checkpoint['G_model'])
                self.optimizerG.load_state_dict(checkpoint['G_optimizer'])
                G_epoch = checkpoint['G_epoch']
        else:
            print(" [*] Failed to find the pretrain_model_G")
            exit()

        # ----------------- load D -------------------
        if not self.pretrain_model_D is None:
            resume_file_D = os.path.join(checkpoint_dir, self.pretrain_model_D)
            flag_D = os.path.isfile(resume_file_D)
            if flag_D == False:
                print('D--> Error: no checkpoint directory found!')
                exit()
            else:
                print('resume_file_D------>: {}'.format(resume_file_D))
                checkpoint = torch.load(resume_file_D)
                self.discriminator1.load_state_dict(checkpoint['D_model1'])
                self.discriminator2.load_state_dict(checkpoint['D_model2'])
                self.discriminator3.load_state_dict(checkpoint['D_model3'])
                self.discriminator4.load_state_dict(checkpoint['D_model4'])
                self.optimizerD1.load_state_dict(checkpoint['D_optimizer1'])
                self.optimizerD2.load_state_dict(checkpoint['D_optimizer2'])
                self.optimizerD3.load_state_dict(checkpoint['D_optimizer3'])
                self.optimizerD4.load_state_dict(checkpoint['D_optimizer4'])
                D_epoch = checkpoint['D_epoch']
        else:
            print(" [*] Failed to find the pretrain_model_D")
            exit()

        print(" [*] Success to load model --> {} & {}".format(self.pretrain_model_G, self.pretrain_model_D))
        return True, G_epoch

    def save(self, checkpoint_dir, index_epoch):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_name = str(index_epoch)+'_'+self.category
        path_save_G = os.path.join(checkpoint_dir, save_name+'_G.pth')
        path_save_D = os.path.join(checkpoint_dir, save_name+'_D.pth')
        print('Save Path for G: {}'.format(path_save_G))
        print('Save Path for D: {}'.format(path_save_D))
        torch.save({
                'G_model': self.generator.state_dict(),
                'G_optimizer': self.optimizerG.state_dict(),
                'G_epoch': index_epoch,
            }, path_save_G)
        torch.save({
                'D_model1': self.discriminator1.state_dict(),
                'D_model2': self.discriminator2.state_dict(),
                'D_model3': self.discriminator3.state_dict(),
                'D_model4': self.discriminator4.state_dict(),
                'D_optimizer1': self.optimizerD1.state_dict(),
                'D_optimizer2': self.optimizerD2.state_dict(),
                'D_optimizer3': self.optimizerD3.state_dict(),
                'D_optimizer4': self.optimizerD4.state_dict(),
                'D_epoch': index_epoch,
            }, path_save_D)

    def MSE_LOSS(self, label, pred):
        return tf.losses.mean_squared_error(label, pred)

################################################################################################
# -------------------------------- class of nework structure -----------------------------------
################################################################################################

# ---------------------------------------G---------------------------------------
import nn_utils

def get_edge_features(x, k, num=-1):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]    
    """
    B, dims, N = x.shape

    # batched pair-wise distance
    xt = x.permute(0, 2, 1)
    xi = -2 * torch.bmm(xt, x)
    xs = torch.sum(xt**2, dim=2, keepdim=True)
    xst = xs.permute(0, 2, 1)
    dist = xi + xs + xst # [B, N, N]

    # get k NN id    
    _, idx_o = torch.sort(dist, dim=2)
    idx = idx_o[: ,: ,1:k+1] # [B, N, k]
    idx = idx.contiguous().view(B, N*k)


    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors) # [B, d, N, k]

    # centralize
    central = x.unsqueeze(3) # [B, d, N, 1]
    central = central.repeat(1, 1, 1, k) # [B, d, N, k]

    ee = torch.cat([central, neighbors-central], dim=1)
    assert ee.shape == (B, 2*dims, N, k)
    return ee

def get_edge_features_xyz(x, pc, k, num=-1):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]
        idx
    """
    B, dims, N = x.shape

    # ----------------------------------------------------------------
    # batched pair-wise distance in feature space maybe is can be changed to coordinate space
    # ----------------------------------------------------------------
    xt = x.permute(0, 2, 1)
    xi = -2 * torch.bmm(xt, x)
    xs = torch.sum(xt**2, dim=2, keepdim=True)
    xst = xs.permute(0, 2, 1)
    dist = xi + xs + xst # [B, N, N]

    # get k NN id    
    _, idx_o = torch.sort(dist, dim=2)
    idx = idx_o[: ,: ,1:k+1] # [B, N, k]
    idx = idx.contiguous().view(B, N*k)

    
    # gather
    neighbors = []
    xyz =[]
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

        tp = torch.index_select(pc[b], 1, idx[b])
        tp = tp.view(3, N, k)
        xyz.append(tp)

    neighbors = torch.stack(neighbors)  # [B, d, N, k]
    xyz = torch.stack(xyz)              # [B, 3, N, k]
    
    # centralize
    central = x.unsqueeze(3).repeat(1, 1, 1, k)         # [B, d, N, 1] -> [B, d, N, k]
    central_xyz = pc.unsqueeze(3).repeat(1, 1, 1, k)    # [B, 3, N, 1] -> [B, 3, N, k]
    
    e_fea = torch.cat([central, neighbors-central], dim=1)
    e_xyz = torch.cat([central_xyz, xyz-central_xyz], dim=1)
    
    assert e_fea.size() == (B, 2*dims, N, k) and e_xyz.size() == (B, 2*3, N, k)
    return e_fea, e_xyz

class conv2dbr(nn.Module):
    """ Conv2d-bn-relu
    [B, Fin, H, W] -> [B, Fout, H, W]
    """
    def __init__(self, Fin, Fout, kernel_size, stride=1):
        super(conv2dbr, self).__init__()
        self.conv = nn.Conv2d(Fin, Fout, kernel_size, stride)
        self.bn = nn.BatchNorm2d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x) # [B, Fout, H, W]
        x = self.bn(x)
        x = self.ac(x)
        return x

class upsample_edgeConv(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, num):
        super(upsample_edgeConv, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.num = num
        
        #self.conv1 = conv2dbr(2*Fin, 2*Fin, 1, 1)
        #self.conv2 = conv2dbr(2*Fin, 2*Fout, [1, 2*k+2], [1, 1])
        self.conv2 = conv2dbr(2*Fin, 2*Fout, [1, 2*k], [1, 1])
        
        self.inte_conv_hk = nn.Sequential(
            #nn.Conv2d(2*Fin, 4*Fin, [1, k//2], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.Conv2d(2*Fin, 4*Fin, [1, k//2+1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(4*Fin),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        B, Fin, N = x.shape
        x = get_edge_features(x, self.k, self.num); # [B, 2Fin, N, k]


        # -------------learn_v2----------------------
        BB, CC, NN, KK = x.size()
        #x = self.conv1(x)
        inte_x = self.inte_conv_hk(x)                                   # Bx2CxNxk/2
        inte_x = inte_x.transpose(2, 1)                                 # BxNx2Cxk/2
        #inte_x = inte_x.contiguous().view(BB, NN, CC, 2, KK//2+1)       # BxNxCx2x(k//2+1)
        #inte_x = inte_x.contiguous().view(BB, NN, CC, KK+2)             # BxNxCx(k+2)
        inte_x = inte_x.contiguous().view(BB, NN, CC, 2, KK//2)         # BxNxCx2x(k//2+1)
        inte_x = inte_x.contiguous().view(BB, NN, CC, KK)               # BxNxCx(k+2)
        inte_x = inte_x.permute(0, 2, 1, 3)                             # BxCxNxk
        merge_x = torch.cat((x, inte_x), 3)                             # BxCxNx2k

        x = self.conv2(merge_x) # [B, 2*Fout, N, 1]

        x = x.unsqueeze(3)                    # BxkcxN
        x = x.contiguous().view(B, self.Fout, 2, N)
        x = x.contiguous().view(B, self.Fout, 2*N)

        assert x.shape == (B, self.Fout, 2*N)
        return x

class bilateral_upsample_edgeConv(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, num, softmax=True):
        super(bilateral_upsample_edgeConv, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.softmax = softmax
        self.num = num

        # self.conv = conv2dbr(2*Fin, Fout, [1, 20], [1, 20])
        #self.conv1 = conv2dbr(2*Fin, 2*Fin, 1 ,1)
        self.conv2 = conv2dbr(2*Fin, 2*Fout, [1, 2*k], [1, 1])

        self.conv_xyz = nn.Sequential(
            nn.Conv2d(6, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_fea = nn.Sequential(
            nn.Conv2d(2*Fin, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_all = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 2*Fin, 1),
            nn.BatchNorm2d(2*Fin),
            nn.LeakyReLU(inplace=True)
        )

        self.inte_conv_hk = nn.Sequential(
            #nn.Conv2d(2*Fin, 4*Fin, [1, k//2], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.Conv2d(2*Fin, 4*Fin, [1, k//2+1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(4*Fin),
            nn.LeakyReLU(inplace = True)
        )

    def forward(self, x, pc):
        B, Fin, N = x.size()
        
        #x = get_edge_features(x, self.k, self.num); # [B, 2Fin, N, k]
        x, y = get_edge_features_xyz(x, pc, self.k, self.num); # feature x: [B, 2Fin, N, k] coordinate y: [B, 6, N, k]
        
        
        w_fea = self.conv_fea(x)
        w_xyz = self.conv_xyz(y)
        w = w_fea * w_xyz
        w = self.conv_all(w)
        if self.softmax == True:
            w = F.softmax(w, dim=-1)    # [B, Fout, N, k] -> [B, Fout, N, k]
        
        # -------------learn_v2----------------------
        BB, CC, NN, KK = x.size()
        #x = self.conv1(x)
        inte_x = self.inte_conv_hk(x)                                   # Bx2CxNxk/2
        inte_x = inte_x.transpose(2, 1)                                 # BxNx2Cxk/2
        inte_x = inte_x.contiguous().view(BB, NN, CC, 2, KK//2)       # BxNxCx2x(k//2+1)
        inte_x = inte_x.contiguous().view(BB, NN, CC, KK)             # BxNxCx(k+2)
      
        inte_x = inte_x.permute(0, 2, 1, 3)                             # BxCxNx(k+2)
        inte_x = inte_x * w

        merge_x = torch.cat((x, inte_x), 3)                             # BxCxNx2k
        
        x = self.conv2(merge_x) # [B, 2*Fout, N, 1]

        x = x.unsqueeze(3)                    # BxkcxN
        x = x.contiguous().view(B, self.Fout, 2, N)
        x = x.contiguous().view(B, self.Fout, 2*N)

        assert x.shape == (B, self.Fout, 2*N)
        return x

class edgeConv(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k):
        super(edgeConv, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv = nn_utils.conv2dbr(2*Fin, Fout, 1)

    def forward(self, x):
        B, Fin, N = x.shape
        x = get_edge_features(x, self.k); # [B, 2Fin, N, k]
        x = self.conv(x) # [B, Fout, N, k]
        x, _ = torch.max(x, 3) # [B, Fout, N]
        assert x.shape == (B, self.Fout, N)
        return x

class bilateral_block_l1(nn.Module):
    def __init__(self,Fin,Fout,maxpool,stride=1,num_k = 20):
        super(bilateral_block_l1,self).__init__()
        self.maxpool = nn.MaxPool2d((1,maxpool),(1,1))
        
        self.upsample_cov = nn.Sequential(
            upsample_edgeConv(Fin, Fout, num_k//2,1),   #(128->256)
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(Fin, 2*Fin),
            #nn.BatchNorm1d(2*Fin),
            #nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        self.g_fc = nn.Sequential(
            nn.Linear(Fout,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self,x):
        batchsize = x.size()[0]
        point_num = x.size()[2]
        xs = self.maxpool(x)
        xs = xs.view(batchsize,-1)
        xs = self.fc(xs)
        
        g = self.g_fc(xs)
        g = g.view(batchsize, -1, 1)
        g = g.repeat(1, 1, 2*point_num)

        xs = xs.view(batchsize,-1,1)
        xs = xs.repeat(1,1,2*point_num)
        x_ec = self.upsample_cov(x)
        x_out = torch.cat((xs,x_ec),1)

        g_out = torch.cat((g, x_ec), dim=1)
        
        return x_out, g_out

class bilateral_block_l2(nn.Module):
    def __init__(self, Fin, Fout, maxpool, stride=1, num_k=20, softmax=True):
        super(bilateral_block_l2,self).__init__()
        self.maxpool = nn.MaxPool2d((1,maxpool),(1,1))
        
        self.upsample_cov = bilateral_upsample_edgeConv(Fin, Fout, num_k//2, 1, softmax=softmax)   #(256->512)
        self.bn_uc = nn.BatchNorm1d(Fout)
        self.relu_uc = nn.LeakyReLU(inplace=True)
        
        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(Fin, 2*Fin),
            #nn.BatchNorm1d(2*Fin),
            #nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        self.g_fc = nn.Sequential(
            nn.Linear(Fout,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x, pc):
        batchsize, _, point_num = x.size()
        xs = self.maxpool(x)
        xs = xs.view(batchsize,-1)
        xs = self.fc(xs)
        
        g = self.g_fc(xs)
        g = g.view(batchsize, -1, 1)
        g = g.repeat(1, 1, 2*point_num)

        xs = xs.view(batchsize,-1,1)
        xs = xs.repeat(1, 1, 2*point_num)

        x_ec = self.relu_uc(self.bn_uc(self.upsample_cov(x, pc)))
        x_out = torch.cat((xs, x_ec), 1)

        g_out = torch.cat((g, x_ec), dim=1)
        
        return x_out, g_out

class bilateral_block_l3(nn.Module):
    def __init__(self, Fin, Fout, maxpool, stride=1, num_k=20, softmax=True):
        super(bilateral_block_l3,self).__init__()
        
        self.maxpool = nn.MaxPool2d((1,maxpool),(1,1))
        
        self.upsample_cov = bilateral_upsample_edgeConv(Fin, Fout, num_k//2, 1, softmax=softmax)   #(256->512)
        self.bn_uc = nn.BatchNorm1d(Fout)
        self.relu_uc = nn.LeakyReLU(inplace=True)
      
        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(Fin,2*Fin),
            #nn.BatchNorm1d(2*Fin),
            #nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        self.g_fc = nn.Sequential(
            nn.Linear(Fout, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, pc):
        batchsize = x.size()[0]
        point_num = x.size()[2]
        xs = self.maxpool(x)
        xs = xs.view(batchsize,-1)
        xs = self.fc(xs)
        
        g = self.g_fc(xs)
        g = g.view(batchsize, -1, 1)
        g = g.repeat(1, 1, 2*point_num)

        xs = xs.view(batchsize,-1,1)
        xs = xs.repeat(1,1,2*point_num)
        #x_ec = self.upsample_cov(x)
        x_ec = self.relu_uc(self.bn_uc(self.upsample_cov(x, pc)))
        x_out = torch.cat((xs,x_ec),1)

        g_out = torch.cat((g, x_ec), dim=1)
        
        return x_out, g_out

class bilateral_block_l4(nn.Module):
    def __init__(self, Fin, Fout, maxpool, stride=1, num_k=20, softmax=True):
        super(bilateral_block_l4, self).__init__()
        
        self.maxpool = nn.MaxPool2d((1,maxpool),(1,1))
        
        self.upsample_cov = bilateral_upsample_edgeConv(Fin, Fout, num_k//2, 1, softmax=softmax)   #(256->512)
        self.bn_uc = nn.BatchNorm1d(Fout)
        self.relu_uc = nn.LeakyReLU(inplace=True)
      
        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(Fin,2*Fin),
            #nn.BatchNorm1d(2*Fin),
            #nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x, pc):
        batchsize = x.size()[0]
        point_num = x.size()[2]
        xs = self.maxpool(x)
        xs = xs.view(batchsize,-1)
        xs = self.fc(xs)
        xs = xs.view(batchsize,-1,1)
        xs = xs.repeat(1,1,2*point_num)
        #x_ec = self.upsample_cov(x)
        x_ec = self.relu_uc(self.bn_uc(self.upsample_cov(x, pc)))
        x_out = torch.cat((xs,x_ec),1)
        
        return x_out

class PointGenerator(nn.Module):
    def __init__(self, num_point=2048, num_k=20, softmax=True):
        super(PointGenerator, self).__init__()
        self.num_point = num_point
        self.num_k = num_k
        self.fc1 = nn.Sequential(
            nn.Linear(128, 4096),
            nn.BatchNorm1d(4096),  # 128,32
            nn.LeakyReLU(inplace=True)
        )
        self.bilateral1 = bilateral_block_l1(32, 32, 128, num_k=num_k)
        self.bilateral2 = bilateral_block_l2(64, 64, 256, num_k=num_k, softmax=softmax)
        self.bilateral3 = bilateral_block_l3(128, 128, 512, num_k=num_k, softmax=softmax)
        self.bilateral4 = bilateral_block_l4(256, 256, 1024, num_k=num_k, softmax=softmax)
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(512+32, 256, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 3, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(512+64, 256, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 3, 1)          
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(512+128, 256, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 3, 1)
        )
        self.mlp4 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.fc1(x)
        x = x.view(batchsize, 32, 128)  # Bx32x128
        
        x1, g_x1 = self.bilateral1(x)           # x1: Bx64x256
        x1s = self.mlp1(g_x1)                   # Bx3x256
        #print('x1: {} x1s: {}'.format(x1.size(), x1s.size()))

        x2, g_x2 = self.bilateral2(x1, x1s)     # x2: Bx128x512
        x2s = self.mlp2(g_x2)                   # Bx3x512
        #print('x2: {} x2s: {}'.format(x2.size(), x2s.size()))
        
        x3, g_x3 = self.bilateral3(x2, x2s)          # x3: Bx256x1024
        x3s = self.mlp3(g_x3)                   # Bx3x1024
        #print('x3: {} x3s: {}'.format(x3.size(), x3s.size()))
        
        x4 = self.bilateral4(x3, x3s)                # x4: Bx512x2048
        x4s = self.mlp4(x4)                     # Bx3x2048
        #print('x4: {} x4s: {}'.format(x4.size(), x4s.size()))
        #exit()
        return x1s, x2s, x3s, x4s

# ---------------------------------------D---------------------------------------


class PointDiscriminator_1(nn.Module):
    def __init__(self, num_point=256):
        super(PointDiscriminator_1, self).__init__()
        self.num_point = num_point
        self.fc1 = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128,256,1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            #nn.Conv1d(256,1024,1),
            #nn.BatchNorm1d(1024),
            #nn.LeakyReLU()
        )
        self.maxpool = nn.MaxPool1d(num_point,1)
        self.mlp = nn.Sequential(
            #nn.Linear(1024,512),
            #nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128,64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64,1)
        )

    def forward(self, x):
        batchsize = x.size()[0]
        #print('d1: x', x.size())
        #x = x.view(batchsize,3,self.num_point)
        x1 = self.fc1(x)
        x2 = self.maxpool(x1)
        x2 = x2.view(batchsize,256)
        x3 = self.mlp(x2)

        return x3

class PointDiscriminator_2(nn.Module):
    def __init__(self, num_point=512):
        super(PointDiscriminator_2, self).__init__()
        self.num_point = num_point
        self.fc1 = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128,256,1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256,512,1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool1d(num_point,1)
        self.mlp = nn.Sequential(
            nn.Linear(512,256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256,64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64,1)
        )

    def forward(self, x):
        batchsize = x.size()[0]
        x1 = self.fc1(x)
        x2 = self.maxpool(x1)
        x2 = x2.view(batchsize,512)
        x3 = self.mlp(x2)

        return x3

class PointDiscriminator_3(nn.Module):
    def __init__(self, num_point=1024):
        super(PointDiscriminator_3, self).__init__()
        self.num_point = num_point
        self.fc1 = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128,256,1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256,512,1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool1d(num_point,1)
        self.mlp = nn.Sequential(
            #nn.Linear(1024,512),
            #nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256,64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64,1)
        )

    def forward(self, x):
        batchsize = x.size()[0]
        #print('d2: x', x.size())
        #x = x.view(batchsize,3,self.num_point)
        x1 = self.fc1(x)
        x2 = self.maxpool(x1)
        x2 = x2.view(batchsize,512)
        x3 = self.mlp(x2)

        return x3

class PointDiscriminator_4(nn.Module):
    def __init__(self, num_point=2048):
        super(PointDiscriminator_4, self).__init__()
        self.num_point = num_point
        self.fc1 = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128,256,1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256,1024,1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool1d(num_point,1)
        self.mlp = nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256,64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64,1)
        )

    def forward(self, x):
        batchsize = x.size()[0]
        #print('d3: x', x.size())
        #x = x.view(batchsize,3,self.num_point)
        x1 = self.fc1(x)
        x2 = self.maxpool(x1)
        x2 = x2.view(batchsize,1024)
        x3 = self.mlp(x2)

        return x3

