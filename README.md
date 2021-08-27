# SegFormer and HRNet Comparason for Semantic Segmentation

 This (incomplete) repo consists of an image segmentation pipeline on the Cityscapes dataset, using [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation), and a powerful new transformer-based architecture called [SegFormer](https://github.com/NVlabs/SegFormer) . The scripts for data preprocessing, training, and inference are done mainly from scratch. The model construction code for HRNet (`models/hrnet.py`) and SegFormer (`models/segformer.py`) have been adapted from the official mmseg implementation, whereas `models/segformer_simple.py` contains a very clean SegFormer implementation that may not be correct. 
 
For training the implementation details of the original papers are followed as closely as possible. 
 
 
#  HRNet

 
 ![](src/stuttgart_hrnet_w48_sample.gif)



# SegFormer



Model Details


##  Official SegFormer
----------------------------------------------------------------------------------------------------

Replication of model in official repository

```python
model = Segformer(
    pretrained=cfg.MODEL.PRETRAINED,
    img_size=1024,
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 6, 40, 3], 
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0, 
    drop_path_rate=0.1,
    decoder_dim = 768
)

```

#### Total Parameters: 85,915,731
----------------------------------------------------------------------------------------------------
#### Total Multiply Adds (For Convolution and Linear Layers only): 11,607 GFLOPs
----------------------------------------------------------------------------------------------------
#### Number of Layers
- Conv2d : 107 layers   
- LayerNorm : 161 layers   
- OverlapPatchEmbed : 4 layers   
- Linear : 264 layers   
- Dropout : 208 layers   
- Attention : 52 layers   
- Identity : 2 layers   
- DWConv : 52 layers   
- GELU : 52 layers   
- Mlp : 52 layers   
- Block : 52 layers   
- DropPath : 102 layers   
- LinearMLP : 4 layers   
- Dropout2d : 1 layers


## Simple SegFormer
----------------------------------------------------------------------------------------------------

```python
model = Segformer(
    dims = (64, 128, 320, 512),     # dimensions of each stage
    heads = (1, 2, 5, 8),           # heads of each stage
    ff_expansion = (4, 4, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    num_layers = (3, 6, 40, 3),     # num layers of each stage
    decoder_dim = 768,              # decoder dimension
    num_classes = 19                # number of segmentation classes
).to(device)
```


#### Total Parameters: 255,280,531

#### Total Multiply Adds (For Convolution and Linear Layers only): 679 GFLOPs

#### Number of Layers
- **MiT** : 1 layers   
- Unfold : 4 layers   
- **Conv2d** : 374 layers   
- **LayerNorm** : 104 layers   
- EfficientSelfAttention : 52 layers   
- **PreNorm** : 104 layers   
- DsConv2d : 52 layers   
- GELU : 52 layers   
- MixFeedForward : 52 layers   
- Upsample : 4 layers

