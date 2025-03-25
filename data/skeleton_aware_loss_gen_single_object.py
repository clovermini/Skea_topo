import os
import sys 
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import gc
import math
import numpy as np
import cv2
import torch
import time
from skimage import measure
from skimage import io
import scipy.ndimage as ndimage
from skimage import morphology
import matplotlib.pyplot as plt
from typing import Callable, Iterable, List, Set, Tuple

from model.utils.utils_overlap import OverlapTile


class SkeletonAwareWeight():
    """
    Skeleton Aware Weight
    
    """
    def __init__(self, eps = 1e-20):
        """
        At this time, the weight is only suited to binary segmentation, so class_num = 2
        """
        self._class_num = 2
        self._eps = eps
        self.method=None
        self.overlap_tile = OverlapTile()  # speed up

    
    def _get_weight(self, mask: np.ndarray, method=None, single_border=False) -> np.ndarray:
        """
        Get skeleton aware weight map
        :param mask: binary gt mask with shape (H, W)
        :return weight:  weight map with shape (H, W, 2)
        """
        self.method = method
        # Get class weight for two channels
        weight = np.zeros((mask.shape[0], mask.shape[1], 2))
        class_weight = np.zeros((self._class_num, 1))
        for class_idx in range(self._class_num):
            idx_num = np.count_nonzero(mask == class_idx)
            class_weight[class_idx, 0] = idx_num
        min_num = np.amin(class_weight)
        class_weight = class_weight * 1.0 / min_num
        class_weight = np.sum(class_weight) - class_weight        
        
        
        # Get weight for each channel
        for class_idx in range(self._class_num):
            temp_mask = np.zeros_like(mask)
            temp_mask[mask == class_idx] = 1.0
            dis_trf = ndimage.distance_transform_edt(temp_mask)
            
            st = time.time()
            if class_idx == 1: 
                # Get weight for border
                if single_border:
                    temp_weight = 1.0
                else:
                    temp_weight = self._get_border_weight(class_weight[class_idx, 0], temp_mask, dis_trf)
            else:
                # Get weight for objects
                label_map, label_num = measure.label(temp_mask, connectivity=1, background=0, return_num=True)
                temp_weight = self._get_object_weight(class_weight[class_idx, 0], dis_trf, label_map, label_num)
            ed = time.time()
            print("class:{}, time {:.4f}".format(class_idx, ed - st))
            weight[:, :, class_idx] = temp_weight * temp_mask
        return weight
    
    def _get_border_weight(self, wc: float, mask: np.ndarray, dis_trf: np.ndarray) -> np.ndarray:
        """
        Get border weight for single connected object
        :param wc: class weight of border channel
        :param mask: real mask with shape (H,W), the border pixels equal 1 and the object pixels equal 0
        :param dis_trf: distance transform of border (shape (H,W)), it means the distance of each border pixel to the nearest object
        :return weight:  weight map of border with shape (H, W)        
        """

        sk = morphology.skeletonize(mask, method="lee") / 255  # Lee Skeleton method
        dis_trf_sk = dis_trf * sk   # Get the distance transform of skeleton pixel

        # Get the distance transform to skeleton pixel
        indices = np.zeros(((np.ndim(sk),) + sk.shape), dtype=np.int32)
        dis_trf_to_sk = ndimage.distance_transform_edt(1 - sk, return_indices=True, indices=indices)

        dis_sk_map = dis_trf_sk[indices[0, :, :], indices[1, :, :]] * mask

        max_dis_trf = np.amax(dis_trf_sk)  # min_dis[i, j, 0] == dis_trf

        weight = 2.0 - ((dis_sk_map + + self._eps) / (max_dis_trf + self._eps))

        weight[weight < 0] = 0.0
        return weight

    def _get_object_weight(self, wc: float, dis_trf: np.ndarray, label_map: np.ndarray, label_num: int) -> np.ndarray:
        """
        Get object weight
        :param wc: class weight of border channel
        :param dis_trf: distance transform of object (shape (H,W)), it means the distance of each pixel to the nearest border
        :param label_map: label map of connected components with shape (H, W)
        :param label_num: the number of connected components
        :return weight:  weight map of object with shape (H, W)        
        """
        weight = np.zeros(label_map.shape)
        image_props = measure.regionprops(label_map, cache=False)
        
        # For each connect component, calculate its weight by its skeleton
        for label_idx in range(label_num):
            image_prop = image_props[label_idx]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0

            bool_sub_sk = morphology.skeletonize(bool_sub, method="lee") /255 # Lee Skeleton method
            if np.count_nonzero(bool_sub_sk == 1.0) == 0:
                # If there is no skelenton pixel, continue
                continue
            # Get the distance transform of skeleton pixel
            dis_trf_sk_sub = dis_trf[min_row: max_row, min_col: max_col] * bool_sub_sk

            # Get the distance transform to skeleton pixel 
            indices = np.zeros(((np.ndim(bool_sub_sk),) + bool_sub_sk.shape), dtype=np.int32)
            dis_trf_to_sk = ndimage.distance_transform_edt(1-bool_sub_sk, return_indices=True, indices=indices)

            h, w = bool_sub.shape[:2]
            dis_sk_map = np.ones((h, w, 2))
            dis_sk_map[:, :, 0] = dis_trf_to_sk  # d0 
            dis_sk_map[:, :, 1] = dis_trf_sk_sub[indices[0, :, :], indices[1, :, :]]   # d1 

            # Rectify, enusre d0 <= d1, d0: the distance of pixel to nearest skeleton pixel, d1: the distance d1 of nearest skeleton pixel to border
            dis_sk_map[:, :, 0][dis_sk_map[:, :, 0] > dis_sk_map[:, :, 1]] = dis_sk_map[:, :, 1][dis_sk_map[:, :, 0] > dis_sk_map[:, :, 1]]    
 
            weight_sub = 1-(dis_sk_map[:, :, 0] / (dis_sk_map[:, :, 1] + self._eps))
            
            weight[min_row: max_row, min_col: max_col] += weight_sub * bool_sub

        
        return weight


if __name__ == "__main__":
    data_dirs = ['DRIVE/']  # noqa
    weight_fuc = SkeletonAwareWeight()
    st = time.time()
    
    for data in data_dirs:
        data_dir = './data/'+data+'labels/'
        label_paths = os.listdir(data_dir)
        print('start to handle .. ', data_dir, ' -- ', len(label_paths))
        cnt = 0
        for i, lbp in enumerate(label_paths):
            if lbp.split('.')[-1] in ['png', 'tif', 'gif']:
                lbp_name = lbp.split('.')[0].strip()
                lbp = os.path.join(data_dir, lbp)
                print('start to handle pic ', lbp, ' -- ', lbp_name)
                img = io.imread(lbp)
                if np.amax(img) == 255 and len(np.unique(img)) == 2:
                    img = img * 1.0 / 255.0

                single_border = False
                if 'iron' in data or 'mass_road' in data:
                    single_border = True   # evenly placed objects
                if 'snemi3d' in data or 'iron' in data:
                    img = 1-img   # If the foreground pixels in the image are 0, then the pixels need to be inverted.
                if 'DRIVE' in data:
                    img = 1.0 - morphology.dilation(img, morphology.square(2))

                weight = weight_fuc._get_weight(img, single_border=single_border)
                print(i, ' costs ', time.time() - st)
                print('weight 0 --> min ', np.amin(weight[:,:,0]), ' max: ', np.amax(weight[:, :, 0]))
                print('weight 1 --> min ', np.amin(weight[:,:,1]), ' max: ', np.amax(weight[:, :, 1]))

                if cnt == 1:
                    plt.figure(figsize=(100, 100))
                    plt.subplot(1,2,1), plt.imshow(weight[:,:,0], cmap='plasma'), plt.title('weight_0')
                    plt.subplot(1,2,2), plt.imshow(weight[:,:,1], cmap='plasma'), plt.title('weight_1')
                    plt.savefig('./weight_'+data.split('/')[0]+'.jpg')
                    plt.show()
                    #break
                
                save_dir = os.path.join(os.path.abspath(os.path.dirname(data_dir)), '../skeaw/')  # noqa
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                np.save(os.path.join(save_dir, lbp_name+'.npy'), weight)
                del weight
                gc.collect()
                cnt += 1
                
        print('handle ... ', data, ' done!')