import pandas as pd
import numpy as np
from PIL import Image
import config
from typing import Tuple, List
from dataclasses import dataclass, field
from config.config import get_config
import logging
import os
from pathlib import Path
from utils.utils import setup_logging

@dataclass
class ROIPreprocessor():
    logger: logging.Logger = field(init=False)
    
    def __post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image to have values between 0 and 1
        """
        return image.astype(np.float32) / 255.0
    
    def normalize_bbox(self, bbox: np.ndarray, image_size: tuple = (512, 512)) -> np.ndarray:
        """
        Normalize bounding box coordinates between 0 and 1
        """
        norm_bbox = bbox.copy()
        self.logger.debug(f"types: {type(norm_bbox[0])}, {type(image_size[0])}")
        norm_bbox[[0, 2]] /= image_size[0]
        norm_bbox[[1, 3]] /= image_size[1]
        
        return norm_bbox
        
    def load_data_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data and labels from the data path and annotation file specified in the config
        """
        config = get_config()
        
        self.logger.info(f"Loading data from {config.data_path}")
        
        try:
            annotation_df = pd.read_csv(config.annotation_path)
        except FileNotFoundError as e:
            self.logger.error(f"Error loading annotation file at path {config.annotation_path}: {str(e)}")
            raise
        
        data: List[np.ndarray] = []
        labels: List[np.ndarray] = []

        total_images = len(annotation_df)
        missed_images = 0
        
        for _, row in annotation_df.iterrows():
            image_path = os.path.join(config.data_path, row["uid_slice"] + '.png')
            try:
                image = Image.open(image_path)
                
            except FileNotFoundError as e:
                missed_images += 1
                continue
            
            image_arr = np.array(image)
            bbox_arr = np.array([row["x"], row["y"], row["width"], row["height"]], dtype=np.float32)
            
            image_norm = self.normalize_image(image_arr)
            bbox_norm = self.normalize_bbox(bbox_arr, image_arr.shape)
            
            data.append(image_norm)
            labels.append(bbox_norm)
            
        if missed_images == total_images:
            self.logger.error("No images loaded, check the data path and annotation file")
            raise ValueError("No images loaded")
        
        data = np.array(data)
        labels = np.array(labels)
        
        self.logger.info(f"Loaded {total_images - missed_images} images, missed {missed_images}")
        self.logger.info(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
        self.logger.info(f"Images and bounding boxes normalized")
        
        return data, labels