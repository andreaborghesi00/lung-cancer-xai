import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy.ndimage as ndimage
import random 


class DLCSNoduleClassificationDataset(Dataset):
    def __init__(self, annotations_df, zoom_factor=0.8, min_size=64, augment:bool=False, shift_limits:tuple=(-5, 5)):
        """
        Args:
            annotations_df (pd.DataFrame): DataFrame containing nodule annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.shift_limits = shift_limits
        self.augment = augment
        self.zoom_factor = zoom_factor
        self.min_size = min_size
        self.hu_min = -1000
        self.hu_max = 500
        self.annotations_df = annotations_df
        self.albumentations_transform = A.Compose([
            # A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.CLAHE(p=1.0),
            # A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])
    
    
    def extract_nodule(self, img, box, perc=None, rand_shift:bool=False, shift_limits:tuple=(-5, 5)):
        """ Extract a nodule from an image given a bounding box, resizing it by a percentage if provided. """
        if perc is not None:
            shift = (random.randint(*shift_limits), random.randint(*shift_limits)) if rand_shift else (0, 0)
            box = self.resize_box(img, box, perc, shift_center=shift)
        nodule = img[box[1]:box[3], box[0]:box[2]]
        return nodule, box
    
    
    def resize_box(self, img, box, percentage, shift_center:tuple=(0,0)):
        """ Resize the bounding box (xyxy format) by a given percentage either by shrinking or enlarging it and by shifting its center , taking care of the image bounds. """
        if not isinstance(box, np.ndarray):
            box = np.array(box)
        box = box.astype(float)
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        box_resized = box.copy()
        box_resized[0] = max(0, box[0] - width * percentage + shift_center[0])
        box_resized[1] = max(0, box[1] - height * percentage + shift_center[1])
        box_resized[2] = min(img.shape[1], box[2] + width * percentage + shift_center[0])
        box_resized[3] = min(img.shape[0], box[3] + height * percentage + shift_center[1])
        
        
        return box_resized.astype(int)


    def upsample_nodule(self, nodule, target_size=(64, 64), order=2, mode='constant'):
        """ Upsample the nodule image to a given target size"""
        if nodule.shape[0] == 0 or nodule.shape[1] == 0:
            raise ValueError("Nodule image is empty, cannot upsample.")
        
        # Calculate the zoom factors
        zoom_factors = (target_size[0] / nodule.shape[0], target_size[1] / nodule.shape[1])
        
        # Upsample the nodule using zoom
        nodule_upsampled = ndimage.zoom(nodule, zoom_factors, order=order, mode=mode)
        
        return nodule_upsampled, zoom_factors
    
    
    def pad_nodule(self, nodule, target_size=(256, 256), mode='constant'):
        """ Pad the nodule image to a target size, keeping the aspect ratio. """
        if nodule.shape[0] == 0 or nodule.shape[1] == 0:
            raise ValueError("Nodule image is empty, cannot pad.")
        
        pad_height = max(0, target_size[0] - nodule.shape[0])
        pad_width = max(0, target_size[1] - nodule.shape[1])
        
        padded_nodule = np.pad(nodule, ((0, pad_height), (0, pad_width)), mode=mode)
        
        return padded_nodule


    def min_upsample(self, nodule, min_short_side=64, order=2, mode='constant'):
        """ Upsample the nodule to ensure its shortest side is at least min_short_side, keeping the aspect ratio. """
        size = nodule.shape
        short_side = min(size)
        if short_side >= min_short_side:
            return nodule, (1.0, 1.0)
        zoom_factor = min_short_side / short_side
        target_size = (int(size[0] * zoom_factor), int(size[1] * zoom_factor))
        nodule_upsampled, _ = self.upsample_nodule(nodule, target_size=target_size, order=order, mode=mode)

        return nodule_upsampled, (zoom_factor, zoom_factor)


    def __len__(self):
        return len(self.annotations_df)


    def __getitem__(self, idx):
        row = self.annotations_df.iloc[idx]
        img_path = row['path']
        img = np.load(img_path)
        bbox = row[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values
        
        nodule, _ = self.extract_nodule(img, bbox, perc=self.zoom_factor, rand_shift=self.augment, shift_limits=self.shift_limits)
        nodule = np.clip(nodule, self.hu_min, self.hu_max)
        nodule = (nodule - self.hu_min) / (self.hu_max - self.hu_min)
        nodule, _ = self.min_upsample(nodule, min_short_side=self.min_size)
        nodule = self.pad_nodule(nodule, target_size=(512, 512))
        nodule = nodule[..., np.newaxis]  # Add channel dimension so that we have the format (H, W, C)
        nodule = np.repeat(nodule, 3, axis=-1)  # Repeat the channel to make it (H, W, 3)
        if self.augment and self.albumentations_transform:
            augmented = self.albumentations_transform(image=nodule)
            nodule = augmented['image']
        else:
            nodule = torch.tensor(nodule, dtype=torch.float32).permute(2, 0, 1)
        
        label = torch.tensor(row['is_benign'], dtype=torch.long)
        
        return nodule, label
    
    def get_loader(self, batch_size=32, shuffle=True, num_workers=4):
        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          )