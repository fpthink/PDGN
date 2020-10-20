## Progressive Point Cloud Deconvolution Generation Network

by Le Hui, Rui Xu, Jin Xie, Jianjun Qian, and  Jian Yang, details are in [paper]( https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600392.pdf).

### Usage

1. requires:

   ```
   CUDA10 + Pytorch 1.2 + Python3
   ```

2. build ops:

   ```
   cd PDGN
   cd lib/pointops && python setup.py install && cd ../../
   ```

3. Dataset:

   ```
   download data: https://github.com/charlesq34/pointnet-autoencoder#download-data
   shapenetcore_partanno_segmentation_benchmark_v0
   ```

4. Train:

   ```
   CUDA_VISIBLE_DEVICES=0,1 python main.py --data_root '/test/dataset/3d_datasets/shapenetcore_partanno_segmentation_benchmark_v0/' --network PDGN_v1 --model_dir PDGN_v1 --batch_size 20 --max_epoch 600 --snapshot 100 --dataset shapenet --choice Chair --phase train
   ```

5. Test:

   ```
   CUDA_VISIBLE_DEVICES=0,1 python main.py --network PDGN_v1 --batch_size 20 --pretrain_model_G 600_Chair_G.pth --pretrain_model_D 600_Chair_D.pth --savename 600_PDGN_v1 --model_dir PDGN_v1 --phase test
   ```



### Citation

If you find the code or trained models useful, please consider citing:

```
@inproceedings{hui2020pdgn,
  title={Progressive Point Cloud Deconvolution Generation Network},
  author={Hui, Le and Xu, Rui and Xie, Jin and Qian, Jianjun and Yang, Jian},
  booktitle={ECCV},
  year={2020}
}
```
