import gc
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb

from sklearn.model_selection import train_test_split

# MONAI imports
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.transforms.box_ops import convert_box_to_mask
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxModed,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    StandardizeEmptyBoxd,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.config import KeysCollection
from monai.data import box_utils
from monai.networks.nets import resnet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    DeleteItemsd,
    apply_transform,
    RandRotated,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    RandZoomd,
    RandFlipd,
    RandRotate90d
)
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.transforms.utility.dictionary import ApplyTransformToPointsd
from monai.utils.type_conversion import convert_data_type

from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch

# local imports
from config.config_3d import get_config
from data.dlcs_dataset import DLCSDataset
from data.dlcs_preprocessing import GenerateBoxMask, GenerateExtendedBoxMask, get_train_transforms, get_val_transforms
import utils.utils as utils
import models.monai_retinanet as rn

logger = logging.getLogger(__name__)
utils.setup_logging()

logger.info(f"Hello from import_test.py")
logger.info(f"current working directory: {os.getcwd()}")

print(f"current working directory: {os.getcwd()}")
# move to lung-cuncer-xai directory
print(f"current working directory: {os.getcwd()}")