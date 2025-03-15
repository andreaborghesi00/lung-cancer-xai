import logging
import utils.utils as utils
from config.config import get_config
import torch
from sklearn.model_selection import train_test_split
import models.faster_rcnn as frcnn
from training.rcnn_trainer import RCNNTrainer
import gc
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import numpy as np
import pandas as pd
from data.dlcs_dataset import DLCSDataset
import os

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.DEBUG)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")    
    logger.info(f"cwd: {os.getcwd()}")
    data_dir = Path("../DLCS/subset_01_processed")
    annotations_path = Path("../DLCS/DLCSD24_Annotations_voxel.csv")
    
    annotations = pd.read_csv(annotations_path)
    logger.info(f"Annotations loaded from {annotations_path}")
    
    # dataset
    ds = DLCSDataset(annotations, data_dir)
    logger.info(f"Dataset initialized with {len(ds)} patients")
    dl = ds.get_loader(shuffle=True)
    
    for data, labels in dl:
        data.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        logger.info(f"Data shape: {data.shape}, Labels shape: {labels['boxes'].shape}")
        