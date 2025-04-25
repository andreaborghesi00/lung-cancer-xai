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
        
        if self.augment:
            self.albu_transforms = A.Compose([
                # Spatial transformations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                
                # Brightness/contrast adjustments (subtle for medical images)
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                
                # Noise and blur (simulates different scan qualities)
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),                
                
                # CT-specific augmentations
                A.CLAHE(clip_limit=2.0, p=0.3),  # Enhances contrast locally,
                
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
        target["labels"] = torch.ones(len(target["boxes"]), dtype=torch.int64) # all boxes are tumors
        image = self._load_image(self.image_paths[idx])
        
        if self.augment:
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