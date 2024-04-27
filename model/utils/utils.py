import os
import cv2
import random
import numpy as np
import torch
from pathlib import Path
from skimage import io
from torch import Tensor
from PIL import Image
import matplotlib.pyplot as plt
from typing import Callable, Iterable, List, Set, Tuple
from IPython.display import clear_output
import matplotlib.pyplot as plt


def setup_seed(seed: int) -> None:
    '''
        Set random seed to make experiments repeatable
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True  # implement same config in cpu and gpu
    torch.backends.cudnn.benchmark = False


def count_classes_ratio(label_dir: Path, obj_value=0) -> Tuple: # (obj_num, bck_num) (1727250, 6137070) for ISBI 2012
    '''
        Calculate classes ratio
        :param label_dir:  Address for labels
        :param target_value: (obj_value, bck_value)
        :return: (obj_num, bck_num, bck_num/obj_num of data)
    '''
    obj_num, bck_num = 0, 0
    
    labels_path = label_dir.glob("*.png")
    for label_path in labels_path:
        img = cv2.imread(str(label_path), 0)
        obj_num += np.count_nonzero(img == obj_value)
        bck_num += np.count_nonzero(img != obj_value)
    return obj_num, bck_num


def count_mean_and_std(img_dir: Path) -> Tuple: # (mean, std, num) (0.495770570343616, 0.16637932545720774, 22) for ISBI 2012 
    '''
        Calculate mean and std of data 
        :param image_path:  Address for images, noted that the value of images should be [0, 1]
        :return: mean, std and num of data
    '''
    assert img_dir.is_dir(), "The input is not a dir"
    mean, std, num = 0, 0, 0

    imgs_path = img_dir.glob("*.png")
    
    for img_path in imgs_path:
        num += 1
        img = cv2.imread(str(img_path), 0) / 255
        assert np.max(np.unique(img)) <= 1, "The img value should lower than 1 when calculate mean and std"
        mean += np.mean(img)
        std += np.std(img)
    mean /= num
    std /= num
    return mean, std, num
    

def load_img(img_path: Path) -> np.ndarray or Image:
    '''
        Load images or npy files
        :param img_path:  Address for images or npy file
        :return: PIL image or numpy array
    '''
    if str(img_path).endswith('.npy'):
        img = np.load(img_path)
    else:
        img = io.imread(str(img_path))
        if np.amax(img) == 255 and len(np.unique(img)) == 2:
            img = img * 1.0 / 255
    return img


class Printer():
    def __init__(self, is_out_log_file=True, file_address=None):
        self.is_out_log_file = is_out_log_file
        self.file_address = file_address
    def print_and_log(self, content, is_print=True):
        if is_print:
            print(content)
        if self.is_out_log_file:
            f = open(os.path.join(self.file_address), "a")
            f.write(content)
            f.write("\n")
            f.close()


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by half every 10 epochs until 1e-5"""
    lr = learning_rate * (0.8 ** (epoch // 10))
    if not lr < 1e-6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# checking function
def check_result(epoch, img, label, output, weight, checkpoint_path, printer, description="val_"): # check the output of dataset
    printer.print_and_log("Image size is {},    min is {}, max is {}".format(img.shape, np.amin(img), np.amax(img)))
    printer.print_and_log("Label size is {},    min is {}, max is {}".format(label.shape, np.amin(label), np.amax(label)))
    printer.print_and_log("Output size is {},   min is {}, max is {}".format(output.shape, np.amin(output), np.amax(output)))
    printer.print_and_log("Weight-0 size is {}, min is {}, max is {}".format(weight[:, :, 0].shape, np.amin(weight[:, :, 0]), np.amax(weight[:, :, 0])))
    printer.print_and_log("Weight-1 size is {}, min is {}, max is {}".format(weight[:, :, 1].shape, np.amin(weight[:, :, 1]), np.amax(weight[:, :, 1])))
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 5, 1), plt.imshow(img), plt.title('img'), plt.axis("off")
    plt.subplot(1, 5, 2), plt.imshow(label, cmap="gray"), plt.title('label'), plt.axis("off")
    plt.subplot(1, 5, 3), plt.imshow(output, cmap="gray"), plt.title('output'), plt.axis("off")
    plt.subplot(1, 5, 4), plt.imshow(weight[:, :, 0], cmap="plasma"), plt.title('weight-0'), plt.axis("off")
    plt.subplot(1, 5, 5), plt.imshow(weight[:, :, 1], cmap="plasma"), plt.title('weight-1'), plt.axis("off")
    plt.savefig(str(Path(checkpoint_path, description + str(epoch).zfill(3) + '_result.png')))
    plt.show()


def plot(epoch, train_value_list, val_value_list, checkpoint_path, find_min_value=True, curve_name='loss'):
    clear_output(True)
    plt.figure()
    target_value = 0
    target_func = 'None'
    if find_min_value and len(val_value_list) > 10:
        target_value = min(val_value_list[10:])
        target_func = 'min'
    elif find_min_value is False and len(val_value_list) > 10:
        target_value = max(val_value_list[10:])
        target_func = 'max'
    title_name = 'Epoch {}. train ' +  curve_name + ': {:.4f}. val ' + curve_name + ': {:.4f}. ' + ' val_' + target_func + ' ' + curve_name + ': {:.4f}. '
    plt.title(title_name.format(epoch, train_value_list[-1], val_value_list[-1], target_value))
    plt.plot(train_value_list, color="r", label="train " + curve_name)
    plt.plot(val_value_list,   color="b", label="val "  + curve_name)
    if len(val_value_list) > 10:
        plt.axvline(x = val_value_list.index(target_value), ls="-",c="green")
        plt.legend(loc="best")
    plt.savefig(str(Path(checkpoint_path,  curve_name + '_curve.png')))
    plt.show()


def uniq(a: Tensor) -> Set:
    """ 获得输入a的unique集合 """
    return set(torch.unique(a.cpu()).numpy())


def is_sset(a: Tensor, sub: Iterable) -> bool:
    """ 判断输入a包含的元素是否为sub的子集 """
    return uniq(a).issubset(sub)


def is_interact(a: List, b: List) -> bool:
    if len( list(set(a).intersection(set(b))) ) == 0:
        return False
    else:
        return True


def file_name_convert(file_list, zfill_num):
    result_list = [str(item).zfill(zfill_num) for item in file_list]
    return result_list