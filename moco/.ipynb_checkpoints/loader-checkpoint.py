# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import torch
import numpy as np
import random
import torchvision.transforms.functional as tf


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class LogTransform(object):
    """
        Return the log of the image
    """
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def __call__(self, x):
        return torch.log(x+self.epsilon)
    
class RandomDropBands(object):
    """
        Drop bands randomly from the image
    """
        
    def __call__(self, x):
        num_bands = np.random.randint(0, x.shape[0])   ## number of bands to keep
        chosen_bands = np.random.choice([*range(x.shape[0])], num_bands, replace=False)
        band_mask = torch.stack([torch.ones(x.shape[1:]) if i in chosen_bands else torch.zeros(x.shape[1:]) for i in range(x.shape[0])])
        masked_image = x * band_mask
        return masked_image.half()