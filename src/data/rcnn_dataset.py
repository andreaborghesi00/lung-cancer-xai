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
    def __init__(self, image_paths: List[str], boxes: torch.Tensor, transform: Optional[Callable] = None, augment: bool = True):
        self.config = get_config()
        self.image_paths = image_paths
        self.boxes = boxes
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)


    def _load_image(self, image_path: str) -> torch.Tensor:
        image = np.load(image_path) # although our ct-scans are 1-channel only, the transforms require 3 channels
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