import os
import sys 
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import time
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path
import torchvision.transforms as tr
from model.utils.utils import *


class WeightMapDataset(Dataset):
    
    def __init__(self, imgs_dir: Path, data_names: List, weight_dir: Path, skelen_dir=None, use_augment=[False, False, False], depth=None, crop_size=None, norm_transform=None, dataset_name=None):
        self._img_path = []
        self._label_path = []
        self._weight_path = []
        if skelen_dir is None:
            self._skelen_path = None
        else:
            self._skelen_path = []
        self.dataset_name = dataset_name
        print('dataloader dataset_name ', self.dataset_name)
        for item in data_names:
            if self.dataset_name == 'mass_road':
                self._img_path.append(Path(imgs_dir, item+'.tiff'))
                self._label_path.append(Path(str(imgs_dir)+'_labels/', item+'.tif'))
            else:
                self._img_path.append(Path(imgs_dir, 'images', item + '.png'))
                self._label_path.append(Path(imgs_dir, 'labels', item + '.png'))
            self._weight_path.append(Path(weight_dir, item + '.npy'))
            if skelen_dir is not None:
                self._skelen_path.append(Path(skelen_dir, item + '.npy'))
        self._use_augment = use_augment
        self._depth = depth
        self._crop_size = crop_size
        self._norm_transform = norm_transform

        img = load_img(self._img_path[0])
        self._h, self._w = img.shape[:2]
        self.cnt_idx = 0
    
    def __getitem__(self, index):
        skelen = None
        # read images
        if self._depth is None: # 2D analysis
            img = load_img(self._img_path[index])  
            label = load_img(self._label_path[index])
            weight = load_img(self._weight_path[index])
            if self._skelen_path is not None:
                skelen = load_img(self._skelen_path[index])
        else:  # 3D analysis
            random_depth = random.randint(0, len(self._img_path) - self._depth)
            img    = np.zeros((self._h, self._w, self._depth))
            label  = np.zeros((self._h, self._w, self._depth))
            weight = np.zeros((self._h, self._w, self._depth, 2))
            for idx, depth_idx in enumerate(range(random_depth, random_depth + self._depth)):
                img[: , :, idx]       = load_img(self._img_path[depth_idx])
                label[: , :, idx]     = load_img(self._label_path[depth_idx])
                weight[: , :, idx, :] = load_img(self._weight_path[depth_idx])
        
        if 'snemi3d' in self.dataset_name or 'iron' in self.dataset_name:
            label = 1.0 - label   # If the foreground in the image is 0, then the pixels need to be inverted.

        # get class weight
        # Get class weight for two channels
        class_num = 2
        class_weight = np.zeros((class_num, 1))
        for class_idx in range(class_num):
            idx_num = np.count_nonzero(label == class_idx)
            class_weight[class_idx, 0] = idx_num
        min_num = np.amin(class_weight)
        class_weight = class_weight * 1.0 / min_num
        class_weight = np.sum(class_weight) - class_weight
        class_weight = torch.from_numpy(class_weight)
        #print('class_Weight ', class_weight)

        # transform
        if self._crop_size is not None:
            img, label, weight, skelen = self._rand_crop(img, label, weight, skelen=skelen, size=self._crop_size)
        if self._use_augment[0]:
            img, label, weight, skelen = self._rand_rotation(img, label, weight, skelen=skelen)
        if self._use_augment[1]:   
            img, label, weight, skelen = self._rand_vertical_flip(img, label, weight, skelen=skelen)    # p<0.5, flip
        if self._use_augment[2]:
            img, label, weight, skelen = self._rand_horizontal_flip(img, label, weight, skelen=skelen)  # p<0.5, flip
        if len(self._use_augment) == 4 and self._use_augment[3] and self._depth is not None:
            img, label, weight, skelen = self._rand_z_filp(img, label, weight, skelen=skelen)           # p<0.5, flip
        

        # to tensor
        if self._depth is None: # 2D analysis->[C,H,W]
            img = self._norm_transform(img)
            label = torch.from_numpy(label[np.newaxis, :, :])
            weight = torch.from_numpy(weight.transpose((2, 0, 1)))
            if skelen is not None:
                skelen = torch.from_numpy(skelen.transpose((2, 0, 1)))
        else: # 3D analysis->[C,D,H,W]
            img    = np.ascontiguousarray(img, dtype=np.float32)
            label  = np.ascontiguousarray(label, dtype=np.float32)
            weight = np.ascontiguousarray(weight, dtype=np.float32)
            img = self._norm_transform(img)
            img = img.unsqueeze(0)
            label = torch.from_numpy(label.transpose((2, 0, 1))[np.newaxis, :, :, :])
            weight = torch.from_numpy(weight.transpose((3, 2, 0, 1))) # shape(H, W, D, C) -> (C, D, H, W)
        item = {'img': img, 'label': label, 'weight': weight, 'class_weight': class_weight}
        if skelen is not None:
            item['skelen'] = skelen
        return item    
    
    def __len__(self):
        return len(self._img_path)

    def _rand_rotation(self, data, mask, weight, skelen=None):
        """
        Random rotation for original image, corresponding label, and others
        :param data:  Original image
        :param mask: Corresponding mask
        :param last: Last information for WPU-Net
        :param weight: Weight map for WPU-Net
        :return:
        """
        # 随机选择旋转角度  Randomly select the rotation angle
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            rotate_idx = 0
        elif angle == 90:
            rotate_idx = 1
        elif angle == 180:
            rotate_idx = 2
        else:
            rotate_idx = 3
        data = np.rot90(data, rotate_idx).copy()
        mask = np.rot90(mask, rotate_idx).copy()
        weight = np.rot90(weight, rotate_idx).copy()
        if skelen is not None:
            skelen = np.rot90(skelen, rotate_idx).copy()
        return data, mask, weight, skelen

    def _rand_vertical_flip(self, data, mask, weight, skelen=None):
        """
        Random vertical flip for original image, corresponding label, and others
        :param data:  Original image
        :param mask: Corresponding mask
        :param last: Last information for WPU-Net
        :param weight: Weight map for WPU-Net
        :return:
        """
        p = random.random()
        if p < 0.5:
            data = np.flipud(data).copy()
            mask = np.flipud(mask).copy()
            weight = np.flipud(weight).copy()
            if skelen is not None:
                skelen = np.flipud(skelen).copy()
        return data, mask, weight, skelen

    def _rand_horizontal_flip(self, data, mask, weight, skelen=None):
        """
        Random horizontal flip for original image, corresponding label, and others
        :param data:  Original image
        :param label: Corresponding label
        :param last: Last information for WPU-Net
        :param weight: Weight map for WPU-Net
        :return:
        """
        p = random.random()
        if p < 0.5:
            data = np.fliplr(data).copy()
            mask = np.fliplr(mask).copy()
            weight = np.fliplr(weight).copy()
            if skelen is not None:
                skelen = np.fliplr(skelen).copy()
        return data, mask, weight, skelen
    
    def _rand_z_filp(self, data, mask, weight, skelen=None):
        """
        Random Z flip for original image, corresponding label, and others
        """
        p = random.random()
        if p < 0.5:
            data = np.flip(data, 2)
            mask = np.flip(mask, 2)
            weight = np.flip(weight, 2)
            if skelen is not None:
                skelen = np.flip(skelen, 2)
        return data, mask, weight, skelen

    def _rand_crop(self, data, mask, weight, skelen=None, size=512):
        """
        Random crop for original image, corresponding label, and others
        :param data:  Original image
        :param mask: Corresponding mask
        :param height: Crop size
        :param width: Crop size
        :param last:  Last information for WPU-Net
        :param weight:  Weight map for WPU-Net
        :return:
        """
        # 随机选择裁剪区域   Randomly select the crop area
        random_h = random.randint(0, data.shape[0] - size)
        random_w = random.randint(0, data.shape[1] - size)

        data = data[random_h: random_h + size, random_w: random_w + size]
        mask = mask[random_h: random_h + size, random_w: random_w + size]
        weight = weight[random_h: random_h + size, random_w: random_w + size]
        if skelen is not None:
            skelen = skelen[random_h: random_h + size, random_w: random_w + size]
        return data, mask, weight, skelen
    