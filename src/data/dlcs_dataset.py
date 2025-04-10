from torch.utils.data import Dataset, DataLoader
import torch
import logging
from utils.utils import setup_logging
from config.config_3d import get_config
from typing import Optional, Callable, List, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from monai.transforms import (
    apply_transform,
    Randomizable
)
from monai.utils import MAX_SEED, get_seed
from monai.data.utils import no_collation
from monai.data import DataLoader as MonaiDataLoader

"""
we expect the annotations to be processed already, i.e. the nodule coordinates are in the same space as the CT,
also the CTs should be already preprocessed and saved as numpy arrays.
NOTE: the given annotations dataframe should contain all the nodules of the listed patients not to incur in data-leakage
"""
class DLCSDataset(Dataset, Randomizable):
    def __init__(
        self,
        annotations: pd.DataFrame,
        data_dir: str,
        transform: Callable,
        image_key: str = "image",
        label_key: str = "label",
        box_key: str = "box", 
    ):
        self.logger = logging.getLogger(__name__)
        self.config = get_config()

        annotations.sort_values(by="patient-id", inplace=True)
        annotations.reset_index(drop=True, inplace=True)

        self.annotations = annotations
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.transform = transform

        self.pids = self.annotations["patient-id"].unique()
        self.logger.info(f"Dataset initialized with {len(self.pids)} patients with a total of {len(self.annotations)} nodules")
        
        self.set_random_state(seed=get_seed())
        self._seed = 0
        
    def __len__(self):
        return len(self.pids)
    
    
    def __getitem__(self, idx):
        pid = self.pids[idx]
        patient_annotations = self.annotations[self.annotations["patient-id"] == pid]
        patient_annotations.reset_index(drop=True, inplace=True)
        filename = f"{self.data_dir}/{pid}.nii.gz"
        
        boxes = torch.tensor(patient_annotations[['coordX', 'coordY', 'coordZ', 'w', 'h', 'd']].values, dtype=torch.float32)
        labels = torch.tensor(patient_annotations["Malignant_lbl"].values, dtype=torch.long)
        labels_zero = torch.zeros_like(labels)
        
        # apply transform
        if isinstance(self.transform, Randomizable):
            self.transform.set_random_state(seed=self._seed)
        data = {
            "image": filename,
            "box": boxes,
            "label": labels_zero,
            }
        
        transformed_data = apply_transform(self.transform, data)
            
        return transformed_data
    
        
    def randomize(self, data: Any = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")
    
    
    def get_loader(self, shuffle: bool = False, num_workers:int=1, batch_size=1):
        return DataLoader(self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=True,
                            collate_fn=no_collation
                            )
    def get_monai_loader(self, shuffle: bool = False, num_workers:int=1, batch_size=1):
        return MonaiDataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=no_collation,
                persistent_workers=True,
            )   