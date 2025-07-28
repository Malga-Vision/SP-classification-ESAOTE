import torch
import numpy as np
from typing import List, Tuple, Union
from torchvision import transforms
import torchvision.transforms.functional as transforms_func
import random


#change brightness and luminance of an image. gamma > 1 makes the shadows darker, while powers smaller than 1 make dark regions lighter. See transforms_func.adjust_gamma
class AdjustGamma:
    """Perform Gamma correction (Power Law transform)."""

    def __init__(self, min_gamma: float = 0.7, max_gamma: float = 1.4,
                 gain: float = 1):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.gain = gain
        # NOTE: the following is actually slightly different from the torchvision adjust_gamma() implementation
        # gamma = tf.random.uniform((1,), min_gamma, max_gamma)
        # img = tf.math.pow(img, gamma)

    def __call__(self, image):
        gamma = random.uniform(self.min_gamma, self.max_gamma)
        return transforms_func.adjust_gamma(image, gamma, self.gain)


class Normalize:
    """Perform normalization and fill NaNs with zeros."""

    def __init__(self):
        pass

    def __call__(self, image):
        lastdim = len(image.size()) - 1
        image /= torch.amax(image, dim=(lastdim-1, lastdim))[..., None, None]
        return torch.where(torch.isnan(image), torch.zeros_like(image), image)


class CustomRotate:
    def __init__(self, rotation_range: int = 10):
        self.rotation_range = rotation_range  # in degrees

    def __call__(self, image):
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        return transforms.functional.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)


class CustomResize:
    """Perform custom resizing."""

    def __init__(self, dsize: Union[int, Tuple[int], List[int]],
                 min_resize_f: float = 0.8, max_resize_f: float = 1.2):
        if isinstance(dsize, int):
            self.dsize = [dsize, dsize]
        elif isinstance(dsize, tuple) or isinstance(dsize, list):
            self.dsize = list(dsize)
            if not all([isinstance(i, int) for i in self.dsize]):
                raise TypeError('All elements in input parameter "dsize" should be integers.')
            if not len(self.dsize) == 2:
                raise TypeError(f'There must be 2 elements in input parameter "dsize", {len(self.dsize)} were found.')
        else:
            raise TypeError('The input parameter "dsize" should be either a list, a tuple or an integer.')
        self.min_resize = min_resize_f
        self.max_resize = max_resize_f
        
    def __call__(self, image):
        size_r = random.uniform(self.min_resize, self.max_resize)
        new_size = [int(size_r * self.dsize[k]) for k in range(2)]     
        return transforms_func.resize(image, new_size, antialias=True)


class ResizeWithCropOrPad:
    """Perform resizing with cropping or padding."""

    def __init__(self, dsize: Union[int, Tuple[int], List[int]]):
        if isinstance(dsize, int):
            self.dsize = [dsize, dsize]
        elif isinstance(dsize, tuple) or isinstance(dsize, list):
            self.dsize = list(dsize)
            if not all([isinstance(i, int) for i in self.dsize]):
                raise TypeError('All elements in input parameter "dsize" should be integers.')
            if not len(self.dsize) == 2:
                raise TypeError(f'There must be 2 elements in input parameter "dsize", {len(self.dsize)} were found.')
        else:
            raise TypeError('The input parameter "dsize" should be either a list, a tuple or an integer.')

    def __call__(self, image):
        imsize = image.size()[-2:]
        pad_size = [0, 0]
        if self.dsize[0] > imsize[0]:
            pad_size[0] = (self.dsize[0] - imsize[0]) // 2 + 1
        if self.dsize[1] > imsize[1]:
            pad_size[1] = (self.dsize[1] - imsize[1]) // 2 + 1
        image = transforms_func.pad(image, pad_size, 0, 'constant')

        return transforms_func.center_crop(image, self.dsize)
    

class RandomHorizontalFlipVideoClip:
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, clip):
        r = random.random()
        if r < self.p:
            transformed_clip = transforms_func.hflip(clip)
            return transformed_clip
        else:
            return clip
        
class RandomVerticalFlipVideoClip:
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, clip):
        r = random.random()
        if r < self.p:
            transformed_clip = transforms_func.vflip(clip)
            return transformed_clip
        else:
            return clip
        



def transforms4us(img_size: Union[int, Tuple[int], List[int]] = 256,
                  gamma_minmax: tuple = (0.7, 1.4), resize_minmax: tuple = (0.8, 1.2),
                  rotation_range: int = 50, num_channels: int=1) -> list:
    
    eco_transforms = [transforms.PILToTensor(),
                      transforms.Resize(img_size, antialias=True),
                      transforms.Grayscale(num_output_channels=num_channels),
                      transforms.ConvertImageDtype(dtype=torch.float),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      AdjustGamma(*gamma_minmax),
                      CustomResize(img_size, *resize_minmax),
                      ResizeWithCropOrPad(img_size),
                      CustomRotate(rotation_range),
                      Normalize()]
    '''
    eco_transforms = [
                      transforms.Resize(img_size, antialias=True),
                      transforms.Grayscale(num_output_channels=num_channels),
                      # Data augmentation transformations
                      transforms.RandomRotation(degrees=10, fill=(0)),
                      transforms.RandomVerticalFlip(p=0.5),            # Random vertical flip
                      transforms.RandomHorizontalFlip(p=0.5),          # Random horizontal flip
                      #transforms.RandomRotation(degrees=10),
                      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=(0)),    #with 3 channels put fill=(0,0,0)  
                      transforms.ToTensor()
                      #transforms.RandomHorizontalFlip(),
                      #transforms.RandomVerticalFlip(),
                      #AdjustGamma(*gamma_minmax),
                      #CustomResize(img_size, *resize_minmax),
                      #ResizeWithCropOrPad(img_size),
                      #CustomRotate(rotation_range),
                      #transforms.Normalize(mean=[0.5], std=[0.5])
                      ]
    '''
    return eco_transforms




def videoClipTransformation(img_size: Union[int, Tuple[int]], resize_minmax: tuple = (0.8, 1.2), 
                            gamma_minmax: tuple = (0.7, 1.4), rotation_range: int = 10, num_channels: int=1):
    
    clip_transforms = [RandomHorizontalFlipVideoClip(), 
                       RandomVerticalFlipVideoClip(),
                       AdjustGamma(*gamma_minmax), 
                       CustomResize(img_size, *resize_minmax), 
                       ResizeWithCropOrPad(img_size), 
                       CustomRotate(rotation_range), 
                       Normalize()
                       ]
    return clip_transforms

 