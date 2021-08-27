import numpy as np
import albumentations
from utils.label_utils import get_labels
from utils.data_utils import label_mapping
from utils.transformations import (normalize, ComposeSingle, ComposeDouble, re_normalize, 
                                   FunctionWrapperSingle, FunctionWrapperDouble, 
                                   AlbuSeg2d, random_crop, random_resize, random_brightness, scale_aug)

labels = get_labels()
id2label =      { label.id      : label for label in labels }


def get_transforms_training(cfg):
    
    transforms_training = ComposeDouble([
        FunctionWrapperDouble(random_resize, scale_factor=16, base_size=cfg.DATASET.BASE_SIZE[1], both=True),
        FunctionWrapperDouble(random_crop, crop_size=cfg.DATASET.CROP_SIZE, ignore_label=cfg.DATASET.IGNORE_LABEL, both=True),
        AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
        FunctionWrapperDouble(label_mapping, label_map=id2label, input=False, target=True),
        FunctionWrapperDouble(random_brightness, input=True, target=False),
        FunctionWrapperDouble(normalize, mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD, input=True, target=False),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    ])
    return transforms_training


def get_transforms_validation(cfg):

    transforms_validation = ComposeDouble([
        FunctionWrapperDouble(random_crop, crop_size=cfg.DATASET.CROP_SIZE, ignore_label=cfg.DATASET.IGNORE_LABEL, both=True),
        FunctionWrapperDouble(label_mapping, label_map=id2label, input=False, target=True),
        FunctionWrapperDouble(normalize, mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD, input=True, target=False),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    ])
    return transforms_validation


def get_transforms_evaluation(cfg):

    transforms_evaluation = ComposeDouble([
        FunctionWrapperDouble(label_mapping, label_map=id2label, input=False, target=True),
        FunctionWrapperDouble(normalize, mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD, input=True, target=False),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    ])
    return transforms_evaluation


def get_transforms_video(cfg):
    transforms_video = ComposeSingle([
        FunctionWrapperSingle(normalize, mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
    ])
    return transforms_video