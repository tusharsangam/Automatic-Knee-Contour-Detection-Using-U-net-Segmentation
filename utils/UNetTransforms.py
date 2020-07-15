import cv2
import numpy as np
from skimage import img_as_float

class Normalize(object):
    def __init__(self, old_range=(0, 1)):
        self.old_range = old_range
    def scale(self, img, old_range, new_range):
        shift = -old_range[0] + new_range[0] * (old_range[1] - old_range[0]) / (new_range[1] - new_range[0])
        scale = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
        return (img + shift) * scale
    def robust_min_max(self, img, consideration_factors=(0.1, 0.1)):
        # sort flattened image
        img_sort = np.sort(img, axis=None)
        # consider x% values
        min_median_index = int(img.size * consideration_factors[0] * 0.5)
        max_median_index = int(img.size * (1 - consideration_factors[1] * 0.5)) - 1
        # return median of highest x% intensity values
        return img_sort[min_median_index], img_sort[max_median_index]
    def __call__(self, sample, out_range=(-1, 1), consideration_factors=(0.1, 0.1)):
        #print(torch.is_tensor(sample["image"]))
        img = sample["image"].astype(np.float32)
        if "label" in sample:
            sample["label"] = sample["label"].astype(np.float32)/255.0
        #img = img/255.0
        
        mean , std = img.mean(), img.std()
        img = (img - mean)/std
        img = np.clip(img, -1.0, 1.0)
        img = (img+1.0)/2.0
        
        sample["image"] = img
        return sample
class ContrastAdjustment(object):
    def __init__(self, clipLimit=2.0, tileGridSize=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    def __call__(self, sample):
        img = sample["image"]
        #print(img.dtype, img.shape)
        sample["image"] = self.clahe.apply(sample["image"])
        return sample