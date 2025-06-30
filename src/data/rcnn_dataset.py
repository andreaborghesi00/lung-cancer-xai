from torch.utils.data import Dataset, DataLoader
import torch
import logging
from utils.utils import setup_logging
from config.config_2d import get_config
from typing import Optional, Callable, List
from PIL import Image
from data.rcnn_preprocessing import ROIPreprocessor
import numpy as np
from torchvision.transforms import v2 as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from PIL import Image
# import torchvision.transforms as T

class StaticRCNNDataset(Dataset):
    def __init__(self, images: torch.Tensor, boxes: torch.Tensor, transform: Optional[Callable] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        self.config = get_config()
        
        self.images = images
        self.boxes = boxes
        self.transform = transform
        self.logger.info(f"Converted images and boxes to tensors with shapes {self.images.shape} and {self.boxes.shape}")
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        target = {
            "boxes": self.boxes[idx].view(-1, 4), # ensures shape (N, 4)
            "labels": torch.ones(len(self.boxes[idx]), dtype=torch.int64) # all boxes are tumors
        }      
        if self.transform:
            image = self.transform(self.images[idx])  
        return image, target
    
    def get_loader(self, shuffle: bool = False, num_workers: int = None, batch_size: int = None):
        return DataLoader(
            self,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.config.dl_workers if num_workers is None else num_workers,
            collate_fn=lambda x: tuple(zip(*x)) # custom collate function to handle the target dictionary
        )
        
class DynamicRCNNDataset(Dataset):
    def __init__(self, image_paths: List[str], boxes: torch.Tensor, transform: Optional[Callable] = None, augment: bool = True):
        self.config = get_config()
        
        self.image_paths = image_paths
        self.boxes = boxes
        self.transform = transform
        self.preprocessor = ROIPreprocessor(transform=self.transform)
        
    def __len__(self):
        return len(self.image_paths)


    def _load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB') # although our ct-scans are 1-channel only, the transforms require 3 channels
        image = np.array(image)
        image = ROIPreprocessor.normalize_background(image)
        if self.transform is None:
            image = self.preprocessor.normalize_image(image)
            image = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(image)
        else:
            # back to PIL image, this is so inefficient, but necessary for the transforms
            image = Image.fromarray(image)
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        target = {
            "boxes": self.boxes[idx].view(-1, 4), # ensures shape (N, 4)
            "labels": torch.ones(len(self.boxes[idx]), dtype=torch.int64) # all boxes are tumors
        }      
        image = self._load_image(self.image_paths[idx])
        return image, target
    
    def get_loader(self, shuffle: bool = False, num_workers: int = None, batch_size: int = None):
        return DataLoader(
            self,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.config.dl_workers if num_workers is None else num_workers,
            collate_fn=lambda x: tuple(zip(*x)), # custom collate function to handle the target dictionary,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True
        )
        
class DynamicResampledNLST(Dataset):
    def __init__(self, image_paths: List[str], boxes: torch.Tensor, augment: bool = True):
        self.config = get_config()
        self.image_paths = image_paths
        self.boxes = boxes
        self.augment= augment
        self.upscale_factor = 1.4
        if self.augment:
            self.albu_transforms = A.Compose([
                # Spatial transformations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1, rotate_limit=10, p=0.5),
                
                # Brightness/contrast adjustments (subtle for medical images)
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                
                # Noise and blur (simulates different scan qualities)
                A.GaussianBlur(blur_limit=(3), p=0.3),
                # A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),                
                
                # CT-specific augmentations
                A.CLAHE(clip_limit=2.0, p=1.0),  # Enhances contrast locally,
                
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.albu_transforms = A.Compose([
                A.CLAHE(clip_limit=2.0, p=1.0),  # Enhances contrast locally,
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    def __len__(self):
        return len(self.image_paths)


    def _load_image(self, image_path: str) -> np.ndarray:
        image = np.load(image_path)     
        return image

    def upscale_numpy_image(img_np, new_size):
        img_pil = Image.fromarray(img_np)
        img_resized = img_pil.resize(new_size, Image.BICUBIC)
        return np.array(img_resized)

    def __getitem__(self, idx):
        target = {
            "boxes": self.boxes[idx].view(-1, 4), # ensures shape (N, 4)
        }      
        target["labels"] = torch.ones(len(target["boxes"]), dtype=torch.int64) # all boxes are tumors TODO: use zeros rather than ones for one-class classification

        image = self._load_image(self.image_paths[idx])
        # image = self.upscale_numpy_image(image, (image.shape[0] * self.upscale_factor, image.shape[1] * self.upscale_factor)) # upscale image by 40%
        boxes_np = target["boxes"].numpy()
        labels_np = target["labels"].numpy()

        transformed = self.albu_transforms(
            image=image,
            bboxes=boxes_np,
            labels=labels_np
        )
        
        image = torch.tensor(transformed["image"], dtype=torch.float32).unsqueeze(0) # add channel dimension
        target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64) 

        return image, target
    
    def get_loader(self, shuffle: bool = False, num_workers: int = None, batch_size: int = None):
        return DataLoader(
            self,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.config.dl_workers if num_workers is None else num_workers,
            collate_fn=lambda x: tuple(zip(*x)), # custom collate function to handle the target dictionary,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True
        )
        
class DynamicResampledDLCS(Dataset):
    def __init__(self, image_paths: np.ndarray, boxes: torch.Tensor, labels:torch.Tensor, augment: bool = True, transform = None):
        self.config = get_config()
        # self.annotations = pd.read_csv(self.config.annotation_file)
        self.data_dir = self.config.data_path
        self.augment = augment
        self.transform = transform
        # self.image_paths = self.annotations["path"].tolist()
        self.image_paths = image_paths
        # self.boxes = self.annotations[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values
        # self.boxes = torch.tensor(self.boxes, dtype=torch.float32)
        self.boxes = boxes
        self.boxes = self.boxes.view(-1, 4) # ensures shape (N, 4)
        
        # self.labels = self.annotations["benign"].values
        # self.labels = self.labels + 1 # shift from 0/1 to 1/2
        # self.labels = torch.tensor(self.labels, dtype=torch.int32)
        self.labels = labels
        self.uniclass_labels = torch.ones(len(self.labels), dtype=torch.int64) # just "nodule"
        
        if self.augment:
            self.albu_transforms = A.Compose([
                # Spatial transformations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                
                # Brightness/contrast adjustments (subtle for medical images)
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                
                # Noise and blur (simulates different scan qualities)
                # A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                # A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),                
                
                # CT-specific augmentations
                A.CLAHE(clip_limit=2.0, p=1.0),  # Enhances contrast locally,                
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        image = np.load(image_path)     
        return image
    
    def __getitem__(self, idx):
        target = {
            "boxes": self.boxes[idx].view(-1, 4), # ensures shape (N, 4)
        }
        target["labels"] = torch.ones(len(target["boxes"]), dtype=torch.int64)
        image = self._load_image(self.image_paths[idx])
        
        # normalize image with bounds between -1000 and 500
        image = np.clip(image, -1000, 500)
        image = (image + 1000) / 1500 # scale to [0, 1]
        
                
        if self.augment:
            boxes_np = target["boxes"].numpy()
            labels_np = target["labels"].numpy()

            transformed = self.albu_transforms(
                image=image,
                bboxes=boxes_np,
                labels=labels_np
            )
            
            image = torch.tensor(transformed["image"], dtype=torch.float32)
            # if self.transform:
            #     image = self.transform(image)
            image = image.unsqueeze(0) # add channel dimension
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64)
            
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        return image, target
    
    def get_loader(self, shuffle: bool = False, num_workers: int = None, batch_size: int = None):
        return DataLoader(
            self,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.config.dl_workers if num_workers is None else num_workers,
            collate_fn=lambda x: tuple(zip(*x)), # custom collate function to handle the target dictionary,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True
        )       
