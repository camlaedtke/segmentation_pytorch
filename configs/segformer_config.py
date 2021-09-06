from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

config = CN()

config.NAME = 'segformer_train_1024x1024_adamw_lr6e-6_wd1e-2_bs_8_epoch400'
config.OUTPUT_DIR = 'outputs'
config.LOG_DIR = 'logs'

config.DATASET = CN()
config.DATASET.NAME = 'cityscapes'
config.DATASET.DATA_DIR = 'data/cityscapes'
config.DATASET.INPUT_PATTERN = '*_leftImg8bit.png'
config.DATASET.ANNOT_PATTERN = '*_gtFine_labelIds.png'
config.DATASET.IMAGE_DIR = 'leftImg8bit'
config.DATASET.LABEL_DIR = 'gtFine'
config.DATASET.NUM_CLASSES = 19
config.DATASET.IGNORE_LABEL = 255
config.DATASET.MEAN = [0.485, 0.456, 0.406]
config.DATASET.STD = [0.229, 0.224, 0.225]
config.DATASET.BASE_SIZE = (1024, 2048)
config.DATASET.CROP_SIZE = (768, 768) # (1024, 1024) # (768, 768)


config.TRAIN = CN()
config.TRAIN.EPOCHS = 400
config.TRAIN.DECAY_STEPS = 160000
config.TRAIN.BATCH_SIZE = 8
config.TRAIN.ACCUM_STEPS = 2
config.TRAIN.ADJ_BATCH_SIZE = config.TRAIN.BATCH_SIZE // config.TRAIN.ACCUM_STEPS
config.TRAIN.POWER = 1.0
config.TRAIN.WARMUP_ITERS = 1500
config.TRAIN.WARMUP_RATIO = 1e-6
config.TRAIN.BY_EPOCH = False
config.TRAIN.BASE_LR = 0.00006
config.TRAIN.MIN_LR = 0.0
config.TRAIN.WARMUP = "linear"
config.TRAIN.OPTIMIZER = 'AdamW'
config.TRAIN.WD = 0.01


config.MODEL = CN()
config.MODEL.NAME = 'segformer'
config.MODEL.PATCH_SIZE = 4


##### B5 #####

config.MODEL.B5 = CN()
config.MODEL.B5.PRETRAINED = 'weights/mit_b5.pth'
config.MODEL.B5.DECODER_DIM  = 768 
config.MODEL.B5.CHANNEL_DIMS = (64, 128, 320, 512) 
config.MODEL.B5.SR_RATIOS    = (8, 4, 2, 1)
config.MODEL.B5.NUM_HEADS    = (1, 2, 5, 8)
config.MODEL.B5.MLP_RATIOS   = (4, 4, 4, 4)
config.MODEL.B5.DEPTHS       = (3, 6, 40, 3)
config.MODEL.B5.QKV_BIAS = True
config.MODEL.B5.DROP_RATE = 0.0
config.MODEL.B5.DROP_PATH_RATE = 0.1

##### B3 #####
config.MODEL.B3 = CN()
config.MODEL.B3.PRETRAINED = 'weights/mit_b3.pth'
config.MODEL.B3.DECODER_DIM  = 768
config.MODEL.B3.CHANNEL_DIMS = (64, 128, 320, 512) 
config.MODEL.B3.SR_RATIOS    = (8, 4, 2, 1)
config.MODEL.B3.NUM_HEADS    = (1, 2, 5, 8)
config.MODEL.B3.MLP_RATIOS   = (4, 4, 4, 4)
config.MODEL.B3.DEPTHS       = (3, 4, 18, 3)
config.MODEL.B3.QKV_BIAS = True
config.MODEL.B3.DROP_RATE = 0.0
config.MODEL.B3.DROP_PATH_RATE = 0.1

##### B2 #####
config.MODEL.B2 = CN()
config.MODEL.B2.PRETRAINED = 'weights/mit_b2.pth'
config.MODEL.B2.DECODER_DIM  = 768
config.MODEL.B2.CHANNEL_DIMS = (64, 128, 320, 512) 
config.MODEL.B2.SR_RATIOS    = (8, 4, 2, 1)
config.MODEL.B2.NUM_HEADS    = (1, 2, 5, 8)
config.MODEL.B2.MLP_RATIOS   = (4, 4, 4, 4)
config.MODEL.B2.DEPTHS       = (3, 4, 6, 3)
config.MODEL.B2.QKV_BIAS = True
config.MODEL.B2.DROP_RATE = 0.0
config.MODEL.B2.DROP_PATH_RATE = 0.1


##### B1 #####
config.MODEL.B1 = CN()
config.MODEL.B1.PRETRAINED = 'weights/mit_b1.pth'
config.MODEL.B1.DECODER_DIM  = 768
config.MODEL.B1.CHANNEL_DIMS = (64, 128, 320, 512) 
config.MODEL.B1.SR_RATIOS    = (8, 4, 2, 1)
config.MODEL.B1.NUM_HEADS    = (1, 2, 5, 8)
config.MODEL.B1.MLP_RATIOS   = (4, 4, 4, 4)
config.MODEL.B1.DEPTHS       = (2, 2, 2, 2)
config.MODEL.B1.QKV_BIAS = True
config.MODEL.B1.DROP_RATE = 0.0
config.MODEL.B1.DROP_PATH_RATE = 0.1


##### B0 #####
config.MODEL.B0 = CN()
config.MODEL.B0.PRETRAINED = 'weights/mit_b0.pth'
config.MODEL.B0.DECODER_DIM  = 256
config.MODEL.B0.CHANNEL_DIMS = (32, 64, 160, 256) 
config.MODEL.B0.SR_RATIOS    = (8, 4, 2, 1)
config.MODEL.B0.NUM_HEADS    = (1, 2, 5, 8)
config.MODEL.B0.MLP_RATIOS   = (4, 4, 4, 4)
config.MODEL.B0.DEPTHS       = (2, 2, 2, 2)
config.MODEL.B0.QKV_BIAS = True
config.MODEL.B0.DROP_RATE = 0.0
config.MODEL.B0.DROP_PATH_RATE = 0.1

