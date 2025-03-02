from torch.utils.data import Dataset, DataLoader
import torch
import logging
from utils.utils import setup_logging
from config.config import get_config
from typing import Optional, Callable, List, Tuple, Union
from PIL import Image
from data.preprocessing import ROIPreprocessor
import numpy as np
from torchvision.transforms import v2 as T

class DynamicTomographyDataset(Dataset):
    def __init__(self, tomography_ids: List[Tuple[str, str]], transform: Optional[Callable] = None):
        """
        tomography ids are in the format (pid, dcm_series)
        boxes are in the format xyxy
        we assume that each tomography has the same box throughout all slices
        """
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging(level=logging.DEBUG)
        
        self.tomography_ids = tomography_ids
        self.transform = transform
        self.preprocessor = ROIPreprocessor(transform=self.transform)
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('L') # grayscale
        image = np.array(image)
        self.logger.debug(f"Loaded image with shape {image.shape}")
        image = ROIPreprocessor.normalize_background(image)
        if self.transform is None:
            image = self.preprocessor.normalize_image(image)
            image = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(image)
        else:
            # back to PIL image, this is so inefficient, but necessary for the transforms
            image = Image.fromarray(image)
            image = self.transform(image)
        return image
    
    def _load_slices_and_boxes(self, tomography_id: Tuple[str, str]) -> torch.Tensor:
        paths, boxes = self.preprocessor.load_paths_labels(tomography_id)
        slices = []
        for path in paths:
            image = self._load_image(path)
            slices.append(image)
        
        slices = torch.stack(slices)
        self.logger.debug(f"Loaded slices with shape {slices.shape}")
        self.logger.debug(f"Loaded boxes with shape {boxes.shape}")
        return slices, boxes
    
    def _compress_labels(self, boxes: torch.Tensor, compression_type: str = "mean") -> torch.Tensor:
        # check if all boxes are the same
        compression_type = compression_type.lower()
        
        if torch.all(torch.eq(boxes, boxes[0])):
            return boxes[0]
        elif compression_type == "union":
            x1 = torch.min(boxes[:, 0])
            y1 = torch.min(boxes[:, 1])
            x2 = torch.max(boxes[:, 2])
            y2 = torch.max(boxes[:, 3])
            return torch.tensor([x1, y1, x2, y2])
        elif compression_type == "mean":
            return torch.mean(boxes, dim=0)
        elif compression_type == "median":
            return torch.median(boxes, dim=0)
        else:
            raise ValueError(f"Invalid compression type {compression_type}")        
        
    def get_loader(self, shuffle: bool = False, num_workers: int = None, batch_size: int = None):
        return DataLoader(
            self,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.config.dl_workers if num_workers is None else num_workers,
            collate_fn=lambda x: tuple(zip(*x)) # custom collate function to handle the target dictionary
        )

    def __len__(self):
        return len(self.tomography_ids)

    
    def __getitem__(self, idx):
        slices, boxes = self._load_slices_and_boxes(self.tomography_ids[idx])
        boxes = self._compress_labels(boxes)
        target = {
            "boxes": boxes.view(-1, 4), # ensures shape (N, 4)
            "labels": torch.ones(len(boxes), dtype=torch.int64) # all boxes are tumors
        }
        return slices, target

# for testing purposes only
if __name__ == "__main__":
    config = get_config()
    preprocessor = ROIPreprocessor()
    tomography_ids = preprocessor.load_tomography_ids()
    tomography_ids = tomography_ids[:2]
    boxes = torch.rand((len(tomography_ids), 4))
    dataset = DynamicTomographyDataset(tomography_ids)
    
    for i in range(len(dataset)):
        dataset[i]