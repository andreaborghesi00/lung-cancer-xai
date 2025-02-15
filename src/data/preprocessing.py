import pandas as pd
import numpy as np
from PIL import Image
import config
from typing import Tuple, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from config.config import get_config
import logging
import os
from pathlib import Path
import utils.utils as utils
from torchvision.transforms import v2 as T
import torch
from multipledispatch import dispatch

@dataclass
class ROIPreprocessor():
    logger: logging.Logger = field(init=False)
    transform: Optional[Callable] = field(default=None)
    
    def __post_init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        utils.setup_logging()
        if self.transform is None:
            self.logger.warning("No transform provided, only basic conversion to tensor and normalization will be applied")
            # self.transform = T.ToImage() # only convert to tensor
        
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
    
    # return tensor like func() -> 
    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load an image from the specified path
        """
        try:
            # as rgb
            image = Image.open(image_path).convert('RGB') # although our ct-scans are 1-channel only, the transforms require 3 channels
            if self.transform is None:
                image = np.array(image)
                image = self.normalize_image(image)
                image = T.ToTensor()(image)
            else:
                image = self.transform(image)
                
        except FileNotFoundError as e:
            raise
        
        return image 
    
    @staticmethod
    def xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
        """
        Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2) format
        """
        bbox_xyxy = bbox.copy()
        bbox_xyxy[2] += bbox[0]
        bbox_xyxy[3] += bbox[1]
        
        return bbox_xyxy
    
    @staticmethod
    def xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
        """
        Convert bounding box from (x1, y1, x2, y2) to (x, y, w, h) format
        """
        bbox_xywh = bbox.copy()
        bbox_xywh[2] -= bbox[0]
        bbox_xywh[3] -= bbox[1]
        
        return bbox_xywh
        
        
    def load_tomography_ids(self) -> List[Tuple[str, str]]:
        """
        Load the unique couples (pid, dcm_series) from the annotation file
        """
        

        self.logger.info(f"Loading data from {self.config.annotation_path}")
        try:
            annotation_df = pd.read_csv(self.config.annotation_path)
        except FileNotFoundError as e:
            self.logger.error(f"Error loading annotation file at path {self.config.annotation_path}: {str(e)}")
            raise
        
        tomography_ids: List[Tuple[str, str]] = []
        for _, row in annotation_df.iterrows():
            tomography_ids.append((row["pid"], row["dcm_series"]))
        
        tot_ids = len(tomography_ids)
        
        # remove duplicates
        tomography_ids = list(set(tomography_ids))
        
        self.logger.info(f"Loaded {len(tomography_ids)} unique tomography ids out of {tot_ids}")
        
        return tomography_ids



    def load_paths_labels(self, tomography_id: Union[Tuple[str, str], List[Tuple[str, str]]]) -> Tuple[List[str], torch.Tensor]:
        """
        Load slice paths and bboxes of the specified tomography id
        """
        try:
            annotation_df = pd.read_csv(self.config.annotation_path)
        except FileNotFoundError as e:
            self.logger.error(f"Error loading annotation file at path {self.config.annotation_path}: {str(e)}")
            raise
        
        if isinstance(tomography_id, tuple):
            tomography_id = [tomography_id]
        
        image_paths = []
        bboxes = []
        missed_paths = 0
        
        for pid, dcm_series in tomography_id:
            for _, row in annotation_df[(annotation_df["pid"] == pid) & (annotation_df["dcm_series"] == dcm_series)].iterrows():
                image_path = os.path.join(self.config.data_path, row["uid_slice"] + '.png')
                if not Path(image_path).exists():
                    missed_paths += 1
                    continue
                
                bbox = torch.tensor([row["x"], row["y"], row["x"] + row["width"], row["y"] + row["height"]], dtype=torch.float32)
                
                image_paths.append(image_path)
                bboxes.append(bbox)
        
        if missed_paths == len(tomography_id):
            self.logger.error("No images loaded, check the data path and annotation file")
            raise ValueError("No images loaded")

        self.logger.info(f"Loaded {len(image_paths)} images, missed {missed_paths}")
        self.logger.info(f"Data shape: {len(image_paths)}, Labels shape: {len(bboxes)}")
        
        bbox_tensor = torch.stack(bboxes)
        return np.array(image_paths), bbox_tensor
        
    # def load_paths_labels(self) -> Tuple[List[str], torch.Tensor]:
    #     """
    #     Load paths and labels from the data path and annotation file specified in the config
    #     """
        
        
    #     self.logger.info(f"Loading data from {self.config.data_path}")
        
    #     try:
    #         annotation_df = pd.read_csv(self.config.annotation_path)
    #     except FileNotFoundError as e:
    #         self.logger.error(f"Error loading annotation file at path {self.config.annotation_path}: {str(e)}")
    #         raise
        
    #     image_paths =[]
    #     bboxes = []

    #     total_images = len(annotation_df)
    #     missed_images = 0
        
    #     for _, row in annotation_df.iterrows():
    #         image_path = os.path.join(self.config.data_path, row["uid_slice"] + '.png')
    #         if not Path(image_path).exists():
    #             missed_images += 1
    #             continue
            
    #         # reads bbox as xyxy directly
    #         bbox = torch.tensor([row["x"], row["y"], row["x"] + row["width"], row["y"] + row["height"]], dtype=torch.float32)
            
    #         image_paths.append(image_path)        
    #         bboxes.append(bbox)
    #     bboxes_tensor = torch.stack(bboxes)
    #     if missed_images == total_images:
    #         self.logger.error("No images loaded, check the data path and annotation file")
    #         raise ValueError("No images loaded")
        
    #     self.logger.info(f"Loaded {total_images - missed_images} images, missed {missed_images}")
    #     self.logger.info(f"Data shape: {len(image_paths)}, Labels shape: {len(bboxes)}")
        
    #     return np.array(image_paths), bboxes_tensor
    
    # def load_data_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Load data and labels from the data path and annotation file specified in the config
    #     """
        
        
    #     self.logger.info(f"Loading data from {self.config.data_path}")
        
    #     try:
    #         annotation_df = pd.read_csv(self.config.annotation_path)
    #     except FileNotFoundError as e:
    #         self.logger.error(f"Error loading annotation file at path {self.config.annotation_path}: {str(e)}")
    #         raise
        
    #     images =[]
    #     bboxes = []

    #     total_images = len(annotation_df)
    #     missed_images = 0
        
    #     for _, row in annotation_df.iterrows():
    #         image_path = os.path.join(self.config.data_path, row["uid_slice"] + '.png')
    #         try:
    #             image = self._load_image(image_path)
                
    #         except FileNotFoundError as e:
    #             missed_images += 1
    #             continue
            
    #         # reads bbox as xyxy directly
    #         bbox = torch.tensor([row["x"], row["y"], row["x"] + row["width"], row["y"] + row["height"]], dtype=torch.float32)
            
    #         images.append(image)
    #         bboxes.append(bbox)
        
    #     if missed_images == total_images:
    #         self.logger.error("No images loaded, check the data path and annotation file")
    #         raise ValueError("No images loaded")
        
    #     images_tensor = torch.stack(images)
    #     del images # trying to survive here
    #     bboxes_tensor = torch.stack(bboxes)
        
    #     self.logger.info(f"Loaded {total_images - missed_images} images, missed {missed_images}")
    #     self.logger.info(f"Data shape: {images_tensor.shape}, Labels shape: {bboxes_tensor.shape}")
    #     self.logger.info(f"Images and bounding boxes normalized")
        
    #     return images_tensor, bboxes_tensor