import cv2
import mmcv
import torch
import random
import numpy as np
import albumentations as A
from typing import List, Callable, Tuple
from skimage.util import crop
from skimage.io import imread
from sklearn.externals._pilutil import bytescale

from utils.data_utils import get_labels
labels = get_labels()
trainid2label = { label.trainId : label for label in labels }


def normalize_01(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (no clipping)"""
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out


def normalize(img: np.ndarray, mean: float, std: float):
    """Normalize based on mean and standard deviation."""
    img = img.astype(np.float32) / 255
    img = img - mean
    img = img / std
    return img


def create_dense_target(tar: np.ndarray):
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    for idx, value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx

    return dummy


def label_mapping(seg: np.ndarray, label_map: dict, inverse=False):
    temp = np.copy(seg)
    if inverse:
        for v, k in label_map.items():
            seg[temp == k] = v
    else:
        for k, v in label_map.items():
            seg[temp == k] = v
    return seg


def cityscapes_label_to_rgb(mask):
    h = mask.shape[0]
    w = mask.shape[1]
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, key in trainid2label.items():
        indices = mask == val
        mask_rgb[indices.squeeze()] = key.color 
    return mask_rgb


def center_crop_to_size(x: np.ndarray, size: Tuple, copy: bool = False) -> np.ndarray:
    """
    Center crops a given array x to the size passed in the function.
    Expects even spatial dimensions!
    """
    x_shape = np.array(x.shape)
    size = np.array(size)
    params_list = ((x_shape - size) / 2).astype(np.int).tolist()
    params_tuple = tuple([(i, i) for i in params_list])
    cropped_image = crop(x, crop_width=params_tuple, copy=copy)
    return cropped_image


def re_normalize(inp: np.ndarray, low: int = 0, high: int = 255):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out


def random_flip(inp: np.ndarray, tar: np.ndarray, ndim_spatial: int):
    flip_dims = [np.random.randint(low=0, high=2) for dim in range(ndim_spatial)]

    flip_dims_inp = tuple([i + 1 for i, element in enumerate(flip_dims) if element == 1])
    flip_dims_tar = tuple([i for i, element in enumerate(flip_dims) if element == 1])

    inp_flipped = np.flip(inp, axis=flip_dims_inp)
    tar_flipped = np.flip(tar, axis=flip_dims_tar)

    return inp_flipped, tar_flipped


def pad_image(img, h, w, size, padvalue):
    pad_image = img.copy()
    pad_h = max(size[0] - h, 0)
    pad_w = max(size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        right = pad_w // 2
        
        if pad_h % 2 == 0: 
            bottom = pad_h // 2
        else:
            bottom = pad_h // 2 + 1
            
        if pad_w % 2 == 0:
            left = pad_w // 2
        else:
            left = pad_w // 2 + 1
        pad_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padvalue)
    return pad_image


def pad_seg(img, h, w, size, padvalue):
    pad_image = img.copy()
    pad_h = max(size[0] - h, 0)
    pad_w = max(size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        
        top = pad_h // 2
        right = pad_w // 2
        
        if pad_h % 2 == 0: 
            bottom = pad_h // 2
        else:
            bottom = pad_h // 2 + 1
            
        if pad_w % 2 == 0:
            left = pad_w // 2
        else:
            left = pad_w // 2 + 1
        # print("--> pad ({},{},{},{})".format(top, bottom, left, right))
        pad_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padvalue)
        pad_image = np.expand_dims(pad_image, 2)
    return pad_image


def random_crop(img, seg, crop_size, ignore_label):
    
    h, w = img.shape[:-1]
    img = pad_image(img, h, w, crop_size, (0.0, 0.0, 0.0))
    seg = pad_seg(seg, h, w, crop_size, (ignore_label,))
    
    if seg.shape[-1] != 1:
        seg = np.expand_dims(seg, -1)
    new_h, new_w = seg.shape[:-1]
    
    x = random.randint(0, new_w - crop_size[1])
    y = random.randint(0, new_h - crop_size[0])
    
    img = img[y:y+crop_size[0], x:x+crop_size[1]]
    seg = seg[y:y+crop_size[0], x:x+crop_size[1]]

    return img, seg



def random_resize(img, seg, scale_factor, base_size, min_scale = 0.5, max_scale=2.0):
    
    rand_scale = 0.5 + random.randint(0, scale_factor) / 10.0
    rand_scale = np.clip(rand_scale, min_scale, max_scale)
    long_size = np.int(base_size * rand_scale + 0.5)
    h, w = img.shape[:2]
    if h > w:
        new_h = long_size
        new_w = np.int(w * long_size / h + 0.5)
    else:
        new_w = long_size
        new_h = np.int(h * long_size / w + 0.5)
    # print(rand_scale, new_h, new_w)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)        
    seg = cv2.resize(seg, (new_w, new_h), interpolation=cv2.INTER_NEAREST)    
    return img, seg


def scale_aug(img, seg=None, scale_factor=1, crop_size=(512, 1024), base_size=(1024, 2048), ignore_label=-1):
        
    img, seg = random_resize(img, seg, scale_factor, base_size)
    
    img, seg = rand_crop(img, seg, crop_size=crop_size, ignore_label=ignore_label)
    
    return img, seg



def random_brightness(img, brightness_shift_value=10):
    if random.random() < 0.5:
        return img
    img = img.astype(np.float32)
    shift = random.randint(-brightness_shift_value, brightness_shift_value)
    img[:, :, :] += shift
    img = np.around(img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class Repr:
    """Evaluable string representation of an object"""

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'


class FunctionWrapperSingle(Repr):
    """A function wrapper that returns a partial for input only."""

    def __init__(self, function: Callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray): return self.function(inp)


class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(self, function: Callable, input: bool = True, target: bool = False, both: bool = False, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target
        self.both = both

    def __call__(self, inp: np.ndarray, tar: dict):
        if self.both:
            inp, tar = self.function(inp, tar)
        else:
            if self.input: inp = self.function(inp)
            if self.target: tar = self.function(tar)
        return inp, tar


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target


class ComposeSingle(Compose):
    """Composes transforms for input only."""

    def __call__(self, inp: np.ndarray):
        for t in self.transforms:
            inp = t(inp)
        return inp


class AlbuSeg2d(Repr):
    """
    Wrapper for albumentations' segmentation-compatible 2D augmentations.
    Wraps an augmentation so it can be used within the provided transform pipeline.
    See https://github.com/albu/albumentations for more information.
    Expected input: (C, spatial_dims)
    Expected target: (spatial_dims) -> No (C)hannel dimension
    """
    def __init__(self, albumentation: Callable):
        self.albumentation = albumentation

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        # input, target
        out_dict = self.albumentation(image=inp, mask=tar)
        input_out = out_dict['image']
        target_out = out_dict['mask']

        return input_out, target_out
