# SegFormer and HRNet Comparason for Semantic Segmentation

This repo consists of an image segmentation pipeline on the Cityscapes dataset, using [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation), and a powerful new transformer-based architecture called [SegFormer](https://github.com/NVlabs/SegFormer) . The scripts for data preprocessing, training, and inference are done mainly from scratch. The model construction code for HRNet (`models/hrnet.py`) and SegFormer (`models/segformer.py`) have been adapted from the official mmseg implementation, whereas `models/segformer_simple.py` contains a very clean SegFormer implementation that may not be correct. 

HRNet and SegFormer are useful architectures to compare, because they represent fundamentally different approaches to image understanding. HRNet - like most other vision architectures - is at its core a series of convolution operations that are stacked, fused, and connected in a very efficient manner. SegFormer, on the other hand, has no convolutional operations, and instead uses transformer layers. It treats each image as a sequence of tokens, where each token represents a 4x4 pixel patch of the image. 

For training, the implementation details of the original papers are followed as closely as possible. 

Due to memory limitations (single RTX 3090 GPU 24 GB), gradient accumilation was used for training the SegFormer model. 


#  HRNet
----------------------------------------------------------------------------------------------------

 ![](src/stuttgart_hrnet_w48_sample.gif)



# SegFormer 
----------------------------------------------------------------------------------------------------


 ![](src/stuttgart_segformer_sample.gif)

