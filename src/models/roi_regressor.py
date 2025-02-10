import torch.nn as nn
import torch.nn.functional as F
from config.config import get_config
from utils.utils import setup_logging
import logging

config = get_config()

class RoiRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        
        self.logger.info("Initializing RoiRegressor")
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # (32, 512, 512)
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # (32, 256, 256)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64, 256, 256)
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # (64, 128, 128)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (128, 128, 128)
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),  # (128, 64, 64)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (256, 64, 64)
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2),  # (256, 32, 32)
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (512, 32, 32)
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(2),  # (512, 16, 16)
        )
        
        self.regressor_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 1024),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.output_dim) # (x, y, w, h)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regressor_head(features)        