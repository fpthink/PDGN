## Progressive Point Cloud Deconvolution Generation Network

by Le Hui, Rui Xu, Jin Xie, Jianjun Qian, and  Jian Yang, details are in [paper]( https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600392.pdf).

### Usage

1. requires:

   ```
   CUDA10.1
   Pytorch 1.7.1
   Python3.7
   ```

2. build ops:

   ```
   cd PDGN
   cd lib/pointops && python setup.py install && cd ../../
   
   cd evaluation/pytorch_structural_losses/
   make clean
   make
   ```

3. Dataset:

   We follow [DPM](https://github.com/luost26/diffusion-point-cloud) and use its processed dataset.
   Please download [shapenet.hdf5](https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ)
   

4. Train:

   ```
   CUDA_VISIBLE_DEVICES=0 python main.py \
      --network PDGNet_v2 \
      --model_dir PDGNet_v2 \
      --batch_size 35 \
      --max_epoch 3000 \
      --snapshot 50 \
      --dataset shapenet15k \
      --choice chair \
      --phase train \
      --data_root dataset/shapenet.hdf5
   ```

5. Test (may take about 2 hours):
   
   ```
   CUDA_VISIBLE_DEVICES=0 python main.py \
      --network PDGNet_v2 \
      --batch_size 50 \
      --pretrain_model_G 600_chair_G.pth \
      --pretrain_model_D 600_chair_D.pth \
      --model_dir PDGNet_v2 \
      --choice chair \
      --phase test
   ```



### Results
1. Results in Chair category (taken from paper [DPM](https://arxiv.org/pdf/2103.01458.pdf)):

   | Model | JSD &#8595; | MMD<br>-CD &#8595; | MMD<br>-EMD &#8595; | COV<br>-CD &#8593; | COV<br>-EMD &#8593; | 1-NNA<br>-CD &#8595; | 1-NNA<br>-EMD &#8595; |
   |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
   | PC-GAN (ICML 18) | 6.649 | 13.436 | 3.104 | 46.23 | 22.14 | 69.67 | 100.00 |
   | GCN-GAN (ICLR 18) | 21.708 | 15.354 | 2.213 | 39.84 | 35.09 | 77.86 | 95.80 |
   | TreeGAN (ICCV 19) | 13.282 | 14.936 | 3.613 | 38.02 | 6.77 | 74.92 | 100.00 |
   | PointFlow (ICCV 19) | 12.474 | 13.631 | 1.856 | 41.86 | 43.38 | 66.13 | 68.40 |
   | ShapeGF (ECCV 20) | 5.996 | 13.175 | 1.785 | 48.53 | 46.71 | 56.17 | 62.69 |
   | **PDGN** (ECCV 20) | 6.764 | 12.852 | 2.082 | 53.48 | 39.33 | 60.71 | 75.53 |
   | DPM (CVPR 21) | 7.797 | 12.276 | 1.784 | 48.94 | 47.52 | 60.11 | 69.06 |

2. Pretrained model in [Chair](https://drive.google.com/drive/folders/1V3NE5Xt__UI4EpgEPcbfb7qVfdalPM_k?usp=sharing) categroy:

   (1) Download and put in path: ./checkpoint/PDGNet_v2/PDGNet_v2
   
   (2) Run the test code.

3. We will provide more pretrained models for other categories soon.

### Citation

If you find the code useful, please consider citing:

```
@inproceedings{hui2020pdgn,
  title={Progressive Point Cloud Deconvolution Generation Network},
  author={Hui, Le and Xu, Rui and Xie, Jin and Qian, Jianjun and Yang, Jian},
  booktitle={ECCV},
  year={2020}
}
```

### Acknowledgement

Our Cuda code is from [PointWeb](https://github.com/hszhao/PointWeb).

Our data processing and evaluation code is from [diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud).
