from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import yaml
import os
from pathlib import Path
import logging

@dataclass(frozen=True)
class ModelConfig:
    project_name: str
    project_dir: str = field(default=os.path.join(Path(__file__).parent.parent.parent), init=False)
    experiment_name: str = field(default=None)
    notes: str = field(default=None)
    use_wandb: bool = False
    
    # Model architecture
    n_input_channels: int = 1
    num_classes: int = 2
    spatial_dims: int = 3
    pretrained: bool = False
    base_anchor_shapes: List[List[int]] = field(default_factory=lambda: [[6,8,4],[8,6,5],[10,10,6]])
    conv1_t_stride: List[int] = field(default_factory=lambda: [2,2,1])
    returned_layers: List[int] = field(default_factory=lambda: [1,2])
    nms_thresh: float = 0.22
    score_thresh: float = 0.02
    spacing: List[float] = field(default_factory=lambda: [0.703125, 0.703125, 1.25])
    
    # Training hyperparameters
    batch_size: int = 4
    epochs: int = 150
    learning_rate: float = 1e-3
    dl_workers: int = 4
    train_split_ratio: float = 0.8
    val_test_split_ratio: float = 0.5
    random_state: int = field(default=None)
    patience: int = 0
    validate_every: int = 5
    warmup_epochs: int = 5
    scaler_init_scale: float = 2 ** 16
    scaler_growth_interval: int = 2000
    
    # Dataset parameters
    patch_size: Tuple[int, int, int] = (192, 192, 72)
    image_key: str = "image"
    box_key: str = "box"
    label_key: str = "label"
    point_key: str = "points"
    label_mask_key: str = "label_mask"
    box_mask_key: str = "box_mask"
    gt_box_mode: str = "cccwhd"
    
    # Paths and Logging
    data_dir: str = "../DLCS/subset_1_to_3_processed"
    annotations_path: str = "../DLCS/DLCSD24_Annotations_voxel_1_to_3.csv"
    last_model_save_path: str = field(default=None)
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints/RetinaNet"
    model_checkpoint: str = field(default=None)
    visualization_dir: str = "visualizations"
    visualization_experiment_name: str = "bbox_and_gradcam"
    
    
    def validate(self):
        def validate(self):
            """Validate configuration parameters with assertions."""
            # General assertions
            assert isinstance(self.project_name, str) and len(self.project_name) > 0, "Project name must be a non-empty string"
            
            # Model architecture assertions
            assert self.n_input_channels > 0, "Number of input channels must be positive"
            assert self.num_classes >= 2, "Number of classes must be at least 2"
            assert self.spatial_dims in [2, 3], "Spatial dimensions must be 2 or 3"
            assert isinstance(self.base_anchor_shapes, list) and len(self.base_anchor_shapes) > 0, "Base anchor shapes must be a non-empty list"
            assert all(isinstance(shape, list) and len(shape) == self.spatial_dims for shape in self.base_anchor_shapes), f"Each anchor shape must be a list of {self.spatial_dims} dimensions"
            assert 0 <= self.nms_thresh <= 1, "NMS threshold must be between 0 and 1"
            assert 0 <= self.score_thresh <= 1, "Score threshold must be between 0 and 1"
            
            # Training hyperparameter assertions
            assert self.batch_size > 0, "Batch size must be positive"
            assert self.epochs > 0, "Number of epochs must be positive"
            assert self.learning_rate > 0, "Learning rate must be positive"
            assert self.dl_workers >= 0, "Number of dataloader workers must be non-negative"
            assert 0 < self.train_split_ratio < 1, "Train split ratio must be between 0 and 1"
            assert 0 < self.val_test_split_ratio < 1, "Validation-test split ratio must be between 0 and 1"
            assert self.patience >= 0, "Patience must be non-negative"
            assert self.validate_every > 0, "Validation frequency must be positive"
            
            # Dataset parameter assertions
            assert all(dim > 0 for dim in self.patch_size), "Patch size dimensions must be positive"
            assert len(self.patch_size) == self.spatial_dims, f"Patch size must have {self.spatial_dims} dimensions"
            assert self.gt_box_mode in ["cccwhd", "xyxyzz", "xyzwhd"], "Invalid ground truth box mode"
            
            # Path assertions
            assert os.path.exists(self.data_dir), f"Data directory does not exist: {self.data_dir}"
            assert os.path.exists(self.annotations_path), f"Annotations file does not exist: {self.annotations_path}"
            
            # If model checkpoint is specified, it should exist
            if self.model_checkpoint is not None:
                assert os.path.exists(self.model_checkpoint), f"Model checkpoint does not exist: {self.model_checkpoint}"
            
            return True
    
    @classmethod
    def load_config(cls, config_path: str = os.path.join(Path(__file__).parent, "config_3d.yaml")) -> "ModelConfig":
        # log current working directory
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at path {config_path}")
        
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

# singleton
_config_instance = None

def get_config(config_path: str = None):
    global _config_instance
    if _config_instance is None:
        if config_path is None:
            config_path = os.path.join(Path(__file__).parent, "config_3d.yaml") # default config path
        _config_instance = ModelConfig.load_config(config_path)
    
    return _config_instance

def reset_config():
    global _config_instance
    _config_instance = None
        