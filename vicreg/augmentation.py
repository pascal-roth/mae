# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# STEP EXPLANATION
# ImageOps.solarize() Invert all pixel values above a threshold
# 


from tkinter import image_types
from PIL import ImageOps, ImageFilter
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# import parameters
from self_sup_seg.third_party.mae.params import IMAGE_MEAN, IMAGE_STD


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(self, 
                 crop_size: int = 224,
                 vic_aug: bool = True
                ) -> None:
        
        self.transform_base = transforms.RandomResizedCrop(
                    crop_size, interpolation=InterpolationMode.BICUBIC
                )
        if vic_aug:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(p=1.0),
                    Solarization(p=0.0),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=IMAGE_MEAN, std=IMAGE_STD
                    ),
                ]
            )
            self.transform_prime = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(p=0.1),
                    Solarization(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=IMAGE_MEAN, std=IMAGE_STD
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=IMAGE_MEAN, std=IMAGE_STD
                    ),
                ]
            )
            self.transform_prime = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=IMAGE_MEAN, std=IMAGE_STD
                    ),
                ]
            )

    def __call__(self, sample):
        # make base transform equally for each image --> manual_seed
        rand_mask_int = torch.randint(low=0, high=10000, size=(1,)) 
        torch.manual_seed(rand_mask_int)
        sample_base = self.transform_base(sample)
        # further augmentation has to be made randomly for each image
        torch.random.seed()
        x1 = self.transform(sample_base)
        x2 = self.transform_prime(sample_base)
        return x1, x2
