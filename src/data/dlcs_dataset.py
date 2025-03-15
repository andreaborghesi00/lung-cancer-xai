from torch.utils.data import Dataset, DataLoader
import torch
import logging
from utils.utils import setup_logging
from config.config import get_config
from typing import Optional, Callable, List, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


"""
we expect the annotations to be processed already, i.e. the nodule coordinates are in the same space as the CT,
also the CTs should be already preprocessed and saved as numpy arrays.
NOTE: the given annotations dataframe should contain all the nodules of the listed patients not to incur in data-leakage
"""
class DLCSDataset(Dataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        data_dir: str,
        transform: Optional[Callable] = None, 
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
        
        
    def __len__(self):
        return len(self.pids)
    
    
    def __getitem__(self, idx):
        pid = self.pids[idx]
        patient_annotations = self.annotations[self.annotations["patient-id"] == pid]
        patient_annotations.reset_index(drop=True, inplace=True)
        
        patient_data = np.load(f"{self.data_dir}/{pid}.npy")
        patient_data = torch.tensor(patient_data, dtype=torch.float32)
        
        # if self.transform is not None:
        #     patient_data = self.transform(patient_data)
        
        labels = self.prepare_labels(patient_annotations, patient_data.shape)
        
        return patient_data, labels
    
    def prepare_labels(self, annotations: pd.DataFrame, shape: Tuple[int, int, int]) -> torch.Tensor:

        box_tensor = torch.tensor(annotations[['coordZ', 'coordX', 'coordY', 'd', 'w', 'h']].values, dtype=torch.float32) / np.repeat(np.array(shape), 2)
        
        labels = {
            "boxes": box_tensor,
            "labels": torch.tensor(annotations["Malignant_lbl"].values, dtype=torch.float32) # 0: malignant, 1: benign
        }
        return labels
    
    def get_loader(self, shuffle: bool = False):
        return DataLoader(self,
                            batch_size=1,
                            shuffle=shuffle,
                            num_workers=1,
                            pin_memory=True)