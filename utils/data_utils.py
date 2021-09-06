from __future__ import print_function, absolute_import, division

import os
import sys
import cv2
import glob
import torch
import pathlib
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
from skimage.io import imread
import matplotlib.pyplot as plt
from utils.label_utils import get_labels
from sklearn.externals._pilutil import bytescale

def re_normalize(inp: np.ndarray, low: int = 0, high: int = 255):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out


labels = get_labels()
id2label =      { label.id      : label for label in labels }
trainid2label = { label.trainId : label for label in labels }

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: dict, split="train", transform=None, labels=True):
        self.cfg = cfg
        self.split = split
        self.labels = labels
        self.crop_size = cfg.CROP_SIZE
        self.base_size = cfg.BASE_SIZE
        
        search_image_files = os.path.join(
            cfg.DATA_DIR,
            cfg.IMAGE_DIR, 
            split, '*', 
            cfg.INPUT_PATTERN)

        if labels:
            search_annot_files = os.path.join(
                cfg.DATA_DIR,
                cfg.LABEL_DIR, 
                split, '*', 
                cfg.ANNOT_PATTERN)
        
        
        # root directory
        root = pathlib.Path.cwd() 

        input_path = str(root / search_image_files)
        if labels:
            target_path = str(root / search_annot_files)
        
        self.inputs = [pathlib.PurePath(file) for file in sorted(glob.glob(search_image_files))]
        if labels:
            self.targets = [pathlib.PurePath(file) for file in sorted(glob.glob(search_annot_files))]
        
        print("Images: {} , Labels: {}".format(len(self.inputs), len(self.targets)))
        
        self.transform = transform
        self.inputs_dtype = torch.float32
        if labels:
            self.targets_dtype = torch.int64
        
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                   1.0166, 0.9969, 0.9754, 1.0489,
                                   0.8786, 1.0023, 0.9539, 0.9843, 
                                   1.1116, 0.9037, 1.0865, 1.0955, 
                                   1.0865, 1.1529, 1.0507]).cuda()
       

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        
        # Select the sample
        input_ID = self.inputs[index]
        if self.labels:
            target_ID = self.targets[index]
        name = os.path.splitext(os.path.basename(input_ID))[0]

        # Load input and target
        if self.labels:
            x, y = imread(str(input_ID)), imread(str(target_ID))
        else:
            x = imread(str(input_ID))
        size = x.shape
            
        # Preprocessing
        if (self.transform is not None) and self.labels:
            x, y = self.transform(x, y)
        elif self.transform is not None:
            x = self.transform(x)

        # Typecasting
        if self.labels:
            x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
            y = y.squeeze()
            return x, y, np.array(size), name
        else:
            x = torch.from_numpy(x).type(self.inputs_dtype)
            return x, np.array(size), name
       
    
    def inference(self, model, image):
        # assume input image is channels first
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        # convert to channels last for resizing
        image = image.numpy()[0].transpose((1,2,0)).copy()
        h, w = self.crop_size
        new_img = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        # convert to channels first for inference
        new_img = new_img.transpose((2, 0, 1))
        new_img = np.expand_dims(new_img, axis=0)
        pred = model(torch.from_numpy(new_img))
        # resize to base size
        pred = F.interpolate(input=pred, size=(ori_height, ori_width), mode='bilinear', align_corners=False)
        # pred = pred.numpy()
        return pred.exp()
    
    
    def label_to_rgb(self, seg):
        h = seg.shape[0]
        w = seg.shape[1]
        seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for key, val in trainid2label.items():
            indices = seg == key
            seg_rgb[indices.squeeze()] = val.color 
        return seg_rgb
    
    
    def save_pred(self, image, pred, sv_path, name):
        pred = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8)
        pred = self.label_to_rgb(pred[0])
        image = image.cpu()
        image = image[0].permute(1,2,0).numpy()
        image = re_normalize(image)

        blend = cv2.addWeighted(image, 0.8, pred, 0.8, 0)
        pil_blend = Image.fromarray(blend).convert("RGB")
        pil_blend.save(os.path.join(sv_path, name[0]+'.png'))

        


def label_mapping(seg: np.ndarray, label_map: dict):
    seg = seg.astype(np.int32)
    temp = np.copy(seg)
    for key, val in label_map.items():
        seg[temp == key] = val.trainId
    return seg


def cityscapes_label_to_rgb(mask):
    h = mask.shape[0]
    w = mask.shape[1]
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for key, val in trainid2label.items():
        indices = mask == key
        mask_rgb[indices.squeeze()] = val.color 
    return mask_rgb


def display(display_list):
    plt.figure(figsize=(15, 5), dpi=150)
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_blend(display_list):
    plt.figure(figsize=(10, 10), dpi=150)
    for i in range(len(display_list)):
        blend = cv2.addWeighted(display_list[i][0], 0.8, display_list[i][1], 0.5, 0)
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(blend)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
