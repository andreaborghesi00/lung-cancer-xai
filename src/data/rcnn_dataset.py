from torch.utils.data import Dataset, DataLoader
import torch
import logging
from utils.utils import setup_logging
from config.config import get_config

class FasterRCNNDataset(Dataset):
    def __init__(self, images, boxes):
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        self.config = get_config()
        
        self.images = torch.tensor(images, dtype=torch.float32)
        self.images = self.images.unsqueeze(1) # add channel dimension
        self.boxes = torch.tensor(boxes, dtype=torch.float32)

        self.logger.info(f"Converted images and boxes to tensors with shapes {self.images.shape} and {self.boxes.shape}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        target = {
            "boxes": self.boxes[idx].view(-1, 4), # ensures shape (N, 4)
            "labels": torch.ones(len(self.boxes[idx]), dtype=torch.int64) # all boxes are tumors
        }        
        return self.images[idx], target
    
    def get_loader(self, shuffle: bool = False, num_workers: int = None, batch_size: int = None):
        return DataLoader(
            self,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.config.dl_workers if num_workers is None else num_workers,
            collate_fn=lambda x: tuple(zip(*x)) # custom collate function to handle the target dictionary
        )