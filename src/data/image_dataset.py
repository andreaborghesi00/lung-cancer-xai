import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from utils.utils import setup_logging

class ImageDataset(Dataset):
    def __init__(self, 
                 data: np.ndarray,
                 annotations: np.ndarray):
        # setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        
        self.data = torch.tensor(data, dtype=torch.float32)
        self.annotations = torch.tensor(annotations, dtype=torch.float32)
        self.logger.info(f"Converted data and annotations to tensors")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.annotations[idx]