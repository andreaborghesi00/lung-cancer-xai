import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from utils.utils import setup_logging
from config.config import get_config

class ImageDataset(Dataset):
    def __init__(self, 
                 data: np.ndarray,
                 annotations: np.ndarray):
        # setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        self.config = get_config()
        
        self.logger.info(f"Converting data and annotations to tensors")
        self.data = torch.tensor(data, dtype=torch.float32)
        self.annotations = torch.tensor(annotations, dtype=torch.float32)
        self.logger.info(f"Converted data and annotations to tensors")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.annotations[idx]
    
    def get_loader(self, shuffle: bool = False, num_workers: int = None, batch_size: int = None):
        return DataLoader(
            self,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.config.dl_workers if num_workers is None else num_workers
        )