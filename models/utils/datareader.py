import os
import json
import torch
import numpy as np
from typing import List
from torchvision import transforms
from torch.utils.data import Dataset 
from torchvision.datasets.folder import default_loader
from .augments import transforms4us, videoClipTransformation, Normalize


def count_samples_per_class(clips, classes):
    target_classes = list(classes.values())
    class_counts_dict = {class_num: 0 for class_num in target_classes}
    for clip in clips:
        with open(clip[0].replace('videos', 'labels')[:-4] + '.txt', 'r') as f:  
            #print("class name", f.read())
            lbl = int(classes[f.read()])
            #print("index label: ", lbl)
        class_counts_dict[lbl] += 1
    return list(class_counts_dict.values())


def samples_per_class_2d(imgs_paths, classes):
    target_classes = list(classes.values())
    class_counts_dict = {class_num: 0 for class_num in target_classes}
    for img in imgs_paths:
        with open(img.replace('videos', 'labels')[:-4] + '.txt', 'r') as f:  
            #print("class name", f.read())
            lbl = int(classes[f.read()])
            #print("index label: ", lbl)
        class_counts_dict[lbl] += 1
    return np.array(list(class_counts_dict.values()))


class EsaoteUSDataset2D(Dataset):

    def __init__(self, root: str, img_paths, split: str, transformations: list = None,
                 in_channels: int = 1, img_size: List[int] = [256, 256], augmentation: bool = False,
                 gamma_minmax: tuple = (0.7, 1.4), resize_minmax: tuple = (0.8, 1.2), rotation_range: int = 10):
        
        self.img_paths = img_paths
        self.split = split
        self.augmentation = augmentation
        
        with open(os.path.join(root, 'classes.json'), 'r') as f:
            self.classes = json.load(f)
        self.num_classes = len(self.classes)
        
        self.samples_per_class = samples_per_class_2d(self.img_paths, self.classes)
        self.num_channels = in_channels
        self.img_size = img_size

        
        self.basic_transforms = transforms.Compose(
            [transforms.PILToTensor(), 
             transforms.Resize(img_size, antialias=True), 
             transforms.Grayscale(num_output_channels=self.num_channels), 
             transforms.ConvertImageDtype(dtype=torch.float), 
             Normalize()]
        )

        if self.split == "train":
            if self.augmentation: 
                print("Applying data augmentation")
                self.transforms = transforms.Compose(
                    transforms4us(img_size=img_size, gamma_minmax=gamma_minmax, resize_minmax=resize_minmax, rotation_range=rotation_range, num_channels=self.num_channels)
                    + (transformations if transformations else [])
                    )
            else:
                print("Not applying data augmentation")
        else:
            self.transforms = None
        
        self.num_samples = len(self.img_paths)

    def __getitem__(self, index):

        if self.split == "train" and self.augmentation:
            img = self.transforms(default_loader(self.img_paths[index]))
        else:
            img = self.basic_transforms(default_loader(self.img_paths[index]))
            
        with open(self.img_paths[index].replace('videos', 'labels')[:-4] + '.txt', 'r') as f: 
            lbl = int(self.classes[f.read()])

        return img, lbl
                              
    def __len__(self):
        return self.num_samples
    
class EsaoteUSDataset2D_comparable_3d(Dataset):

    def __init__(self, root: str, img_paths, split: str, transformations: list = None,
                 in_channels: int = 1, img_size: List[int] = [256, 256], augmentation: bool = False,
                 gamma_minmax: tuple = (0.7, 1.4), resize_minmax: tuple = (0.8, 1.2), rotation_range: int = 10):
        
        self.img_paths = img_paths
        self.split = split
        self.augmentation = augmentation
        
        with open(os.path.join(root, 'classes.json'), 'r') as f:
            self.classes = json.load(f)
        self.num_classes = len(self.classes)
        
        self.samples_per_class = samples_per_class_2d(self.img_paths, self.classes)  

        self.num_channels = in_channels
        self.img_size = img_size


        
        self.basic_transforms = transforms.Compose(
            [transforms.PILToTensor(), 
             transforms.Resize(img_size, antialias=True), 
             transforms.Grayscale(num_output_channels=self.num_channels), 
             transforms.ConvertImageDtype(dtype=torch.float), 
             Normalize()]
        )
        
        if self.split == "train":
            if self.augmentation: 
                print("Applying data augmentation")
                self.transforms = transforms.Compose(
                    transforms4us(img_size=img_size, gamma_minmax=gamma_minmax, resize_minmax=resize_minmax, rotation_range=rotation_range, num_channels=self.num_channels)
                    + (transformations if transformations else [])
                    )
            else:
                print("Not applying data augmentation")
        else:
            self.transforms = None
        
        self.num_samples = len(self.img_paths)

    def __getitem__(self, index):

        if self.split == "train" and self.augmentation:
            img = self.transforms(default_loader(self.img_paths[index]))
        else:
            img = self.basic_transforms(default_loader(self.img_paths[index]))

        with open(self.img_paths[index].replace('videos', 'labels')[:-4] + '.txt', 'r') as f: 
            lbl = int(self.classes[f.read()])
        

        return img, lbl
                              
    def __len__(self):
        return self.num_samples




class EsaoteUSDataset3D(Dataset):
    def __init__(self, root: str, clips_paths, split: str, transformations: list = None,
                 in_channels: int = 1, img_size: List[int] = [256, 256],
                 gamma_minmax: tuple = (0.7, 1.4), resize_minmax: tuple = (0.8, 1.2), rotation_range: int = 10,
                 augmentation: bool = False):   
        self.clips_paths = clips_paths  
             
        self.split = split
        self.augmentation = augmentation
        self.num_channels = in_channels
        self.img_size = img_size
        
        with open(os.path.join(root, 'classes.json'), 'r') as f:
            self.classes = json.load(f)

        self.num_classes = len(self.classes)
        self.clip_len = len(self.clips_paths[0])
        self.num_clips = len(self.clips_paths)

        self.samples_per_class = np.array(count_samples_per_class(self.clips_paths, self.classes))

        self.basic_transforms = transforms.Compose(
            [transforms.PILToTensor(), 
            transforms.Grayscale(num_output_channels=self.num_channels), 
            transforms.ConvertImageDtype(dtype=torch.float), 
            transforms.Resize(img_size, antialias=True)]
        )

        self.norm = transforms.Compose([Normalize()])
        
        if self.split == "train":
            if self.augmentation:
                self.transforms = transforms.Compose(
                    videoClipTransformation(img_size)
                    )
        else:
            self.transforms = None
        

    def __getitem__(self, index):
        clip_path = self.clips_paths[index]
        clip_frames = torch.zeros([self.clip_len, self.num_channels, *self.img_size], dtype=torch.float)

        for idx, frame_path in enumerate(clip_path):
            clip_frames[idx] = self.basic_transforms(default_loader(frame_path))
        if self.split == 'train' and self.augmentation:
            clip_frames = self.transforms(clip_frames)
        else:
            clip_frames = self.norm(clip_frames)
        with open(clip_path[0].replace('videos', 'labels')[:-4] + '.txt', 'r') as f:       
            lbl = int(self.classes[f.read()])
        return clip_frames, lbl
    
    def __len__(self):
        return self.num_clips
    
