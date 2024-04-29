import math
import numpy as np
from typing import Callable, Iterable, List, Set, Tuple, Dict


class OverlapTile:
    """
        This stategy is implementated from: Ma B, Ban X, Huang H, et al. Deep learning-based image segmentation
        for al-la alloy microscopic images[J]. Symmetry, 2018, 10(4): 107.
        For figure 4
        crop_size is the size of blue rectangle, which is equal to input_size
        roi_size is the size of yellow rectangle
    """

    def __init__(self, crop_size: int = 256, overlap_size: int = 32):
        self._crop_size = crop_size
        self._overlap_size = overlap_size
        self._roi_size = crop_size - 2 * overlap_size
        self._in_img_shape = None

    def crop(self, in_img: np.ndarray) -> List:
        """
            Crop in_img to sub-img List,
            hint: self._roi_size is used in crop and stitch stage, please check the paper careful when read this code
        """
        self._in_img_shape = in_img.shape
        # Pad image before cropping
        in_pad_img = np.pad(in_img, self._overlap_size, mode='symmetric')
        # calculate the number of cropping, which consider the influence of remainder
        h_pad_num = math.ceil(in_pad_img.shape[0] / self._roi_size)
        w_pad_num = math.ceil(in_pad_img.shape[1] / self._roi_size)
        # if in_pad_img.shape[0] % (self._roi_size) == 0:h_pad_num = h_pad_num - 1
        # if in_pad_img.shape[1] % (self._roi_size) == 0:w_pad_num = w_pad_num - 1
        in_crop_imgs = []
        for i in range(h_pad_num):
            # row analysis
            # overlap_crop, it is need to calculate the start of cropping
            start_h = i * self._roi_size
            end_h = start_h + self._crop_size

            # if there is some remainder result for cropping, change start_h and start_w
            if end_h > in_pad_img.shape[0]:
                start_h = in_pad_img.shape[0] - self._crop_size
                end_h = in_pad_img.shape[0]

            for j in range(w_pad_num):
                # column analysis
                start_w = j * self._roi_size
                end_w = start_w + self._crop_size

                if end_w > in_pad_img.shape[1]:
                    start_w = in_pad_img.shape[1] - self._crop_size
                    end_w = in_pad_img.shape[1]

                crop_img = in_pad_img[start_h: end_h, start_w: end_w]
                # print("cropping: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}, crop_shape={}".\
                #      format(i, start_h, start_w, j, end_h, end_w, crop_img.shape))
                in_crop_imgs.append(crop_img)
                if end_w == in_pad_img.shape[1]:
                    break
            if end_h == in_pad_img.shape[0]:
                break
        return in_crop_imgs

    def stitch(self, out_crop_imgs: List) -> np.ndarray:
        """
            Stitch sub-img List to whole out img
        """
        out_img = np.zeros(self._in_img_shape)

        # calculate the number of cropping, which consider the influence of remainder
        # h_num = math.ceil(self._in_img_shape[0] / self._roi_size) # it is no need to use that
        w_num = math.ceil(self._in_img_shape[1] / self._roi_size)

        for img_idx, out_crop_img in enumerate(out_crop_imgs):
            roi_img = out_crop_img[self._overlap_size: self._overlap_size + self._roi_size,
                      self._overlap_size: self._overlap_size + self._roi_size]
            i = int(img_idx / w_num)
            j = int(img_idx - (i * w_num))
            start_h = int(i * self._roi_size);
            end_h = start_h + self._roi_size
            start_w = int(j * self._roi_size);
            end_w = start_w + self._roi_size

            if end_h > self._in_img_shape[0]:
                start_h = self._in_img_shape[0] - self._roi_size
                end_h = self._in_img_shape[0]
            if end_w > self._in_img_shape[1]:
                start_w = self._in_img_shape[1] - self._roi_size
                end_w = self._in_img_shape[1]
                # print("stitching: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}"\
            # .format(i, start_h, start_w, j, end_h, end_w))
            out_img[start_h: end_h, start_w: end_w] = roi_img
        return out_img