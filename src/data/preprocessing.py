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
# import torchvision.transforms as T
import torch
from tqdm import tqdm

@dataclass
class ROIPreprocessor():
    logger: logging.Logger = field(init=False)
    transform: Optional[Callable] = field(default=None)
    
    def __post_init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        utils.setup_logging(level=logging.DEBUG)
        if self.transform is None:
            self.logger.warning("No transform provided, only basic conversion to tensor and normalization will be applied")
            # self.transform = T.ToImage() # only convert to tensor
            
        try:
            self.annotation_df = pd.read_csv(self.config.annotation_path)
        except FileNotFoundError as e:
            self.logger.error(f"Error loading annotation file at path {self.config.annotation_path}: {str(e)}")
            raise
        
        if not all(key in self.annotation_df.columns for key in ["pid", "dcm_series", "uid_slice", "x", "y", "width", "height"]):
            self.logger.error("Annotation file must contain the following columns: pid, dcm_series, uid_slice, x, y, width, height")
            raise ValueError("Invalid annotation file")
    
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
    def xywh_to_yolo(bbox: np.ndarray) -> np.ndarray:
        """
        Convert bounding box from (x, y, w, h) to (x_center, y_center, w, h) format
        """
        bbox_yolo = bbox.copy()
        bbox_yolo[0] += bbox[2] / 2
        bbox_yolo[1] += bbox[3] / 2
        
        return bbox_yolo
        
    @staticmethod
    def xyxy_to_yolo(bbox: np.ndarray) -> np.ndarray:
        """
        Generate bounding box from (x1, y1, x2, y2) to (x_center, y_center, w, h) format
        """
        bbox_yolo = bbox.copy()
        bbox_yolo[0] += (bbox[2] - bbox[0]) / 2
        bbox_yolo[1] += (bbox[3] - bbox[1]) / 2
        bbox_yolo[2] = bbox[2] - bbox[0]
        bbox_yolo[3] = bbox[3] - bbox[1]
        
        return bbox_yolo
    
    
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
        tomography_ids: List[Tuple[str, str]] = []
        for _, row in self.annotation_df.iterrows():
            tomography_ids.append((row["pid"], row["dcm_series"]))
        
        tot_ids = len(tomography_ids)
        
        # remove duplicates
        tomography_ids = list(set(tomography_ids))
        # sort by pid
        tomography_ids.sort(key=lambda x: x[1].split('.')[-1]) # this is to ensure reproducibility, as set does not guarantee order, the splitting will be different even if the seed is the same
        
        self.logger.info(f"Loaded {len(tomography_ids)} unique tomography ids out of {tot_ids}")
        
        # filter out the tomographies that correspond to a non-existing slice path
        available_slices = [os.path.splitext(f)[0] for f in os.listdir(self.config.data_path)]
        self.logger.debug(f"File format that i am looking for: {available_slices[0]}")
        
        valid_tomography_ids = []
        
        for pid, dcm_series in tomography_ids:
            for _, row in self.annotation_df[(self.annotation_df["pid"] == pid) & (self.annotation_df["dcm_series"] == dcm_series)].iterrows():
                if row["uid_slice"] in available_slices:
                    valid_tomography_ids.append((pid, dcm_series))
                    break
        
        self.logger.info(f"Found {len(valid_tomography_ids)} valid tomography ids out of {len(tomography_ids)}")
        
        return valid_tomography_ids

    @staticmethod
    def normalize_background(image: Union[np.ndarray, torch.Tensor], new_bg_value: int = 255) -> Union[np.ndarray, torch.Tensor]:
        """
        Finds the most common pixel value in the image and sets it as the background to 255
        """
        if new_bg_value < 0 or new_bg_value > 255:
            raise ValueError("New value must be between 0 and 255")
        
        if isinstance(image, torch.Tensor):
            background = torch.bincount(image.flatten().long()).argmax()
            image[image == background] = new_bg_value
            return image
        elif isinstance(image, np.ndarray):    
            background = np.bincount(image.flatten()).argmax()
            image[image == background] = new_bg_value
            return image
        
        raise ValueError("Input must be a numpy array or a torch tensor")
        
    def load_paths_labels(self, tomography_id: Union[Tuple[str, str], List[Tuple[str, str]]]) -> Tuple[List[str], torch.Tensor]:
        """
        Load slice paths and bboxes (xyxy format) of the specified tomography id (pid, dcm_series)
        """
        if isinstance(tomography_id, tuple):
            tomography_id = [tomography_id]
        
        image_paths = []
        bboxes = []
        missed_paths = 0
        for pid, dcm_series in tomography_id:
            for _, row in self.annotation_df[(self.annotation_df["pid"] == pid) & (self.annotation_df["dcm_series"] == dcm_series)].iterrows():
                image_path = os.path.join(self.config.data_path, row["uid_slice"] + '.png')
                self.logger.debug(f"Loading image at path {image_path}")
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