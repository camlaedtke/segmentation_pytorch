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
config.DATASET.CROP_SIZE = (1024, 1024)


config.TRAIN = CN()
config.TRAIN.EPOCHS = 400
config.TRAIN.DECAY_STEPS = 160000
config.TRAIN.BATCH_SIZE = 8
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
config.MODEL.PRETRAINED = 'weights/mit_b5.pth'
config.MODEL.DECODER_DIM = 768 # B5
# config.MODEL.DECODER_DIM = 256 # B0
# config.MODEL.CHANNEL_DIMS = (32, 64, 160, 256) # B0
config.MODEL.CHANNEL_DIMS = (64, 128, 320, 512) # B5 
config.MODEL.FEATURE_STRIDES = (4, 8, 16, 32) 