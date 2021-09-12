# MGTnet: Image-Guided Human Reconstruction via Multi-Scale Graph Transformation Networks
This repository is the offical tensorflow implementation of [MGTnet:Image-Guided Human Reconstruction via Multi-Scale Graph Transformation Networks(TIP 2020).](http://cic.tju.edu.cn/faculty/likun/projects/MGTnet/index.html)

![fig1](https://github.com/1020244018/MGTnet/blob/main/assert/fig1.jpg)
## Introduction
3D human reconstruction from a single image is a challenging problem. Existing methods have difficulties to infer 3D clothed human models with consistent topologies for various poses. In this paper, we propose an efficient and effective method using a hierarchical graph transformation network. To deal with large deformations and avoid distorted geometries, rather than using Euclidean coordinates directly, 3D human shapes are represented by a vertex-based deformation representation that effectively encodes the deformation and copes well with large deformations. To infer a 3D human mesh consistent with the input real image, we also use a perspective projection layer to incorporate perceptual image features into the deformation representation. Our model is easy to train and fast to converge with short test time.


## D^2Human Dataset
We present the D^2Human (Dynamic Detailed Human) dataset, including variously posed 3D human meshes with consistent topologies and rich geometry details, together with the captured color images and SMPL.
![dataset](https://github.com/1020244018/MGTnet/blob/main/assert/datasets.jpg)
[BaiDuYunPan Download](https://pan.baidu.com/s/1A7kvSWhu0sHUh8p6-htQOw) with extraction code 69ba.

## Citation
Please considering citing 
```bibtex
@article{li2021image,
  title={Image-Guided Human Reconstruction via Multi-Scale Graph Transformation Networks},
  author={Li, Kun and Wen, Hao and Feng, Qiao and Zhang, Yuxiang and Li, Xiongzheng and Huang, Jing and Yuan, Cunkuan and Lai, Yu-Kun and Liu, Yebin},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={5239--5251},
  year={2021},
  publisher={IEEE}
}
```
