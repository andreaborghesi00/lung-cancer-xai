from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml
import os
from pathlib import Path
import logging

@dataclass(frozen=True)
class ModelConfig:
    experiment_name: str
    project_dir: str = field(default=os.path.join(Path(__file__).parent.parent.parent), init=False)
    
    # Model architecture
    image_input_channels: int = 1
    num_classes: int = 2
    output_dim: int = 4
    
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    dropout_rate: float = 0.3

    # Paths and Logging
    data_path: str = "data/slices_segmentate_png"
    annotation_path: str = "data/annotations.csv"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    def validate(self):
        assert self.image_input_channels > 0, "Image input channels must be greater than 0"
        assert self.num_classes > 0, "Number of classes must be greater than 0"
        assert self.batch_size > 0, "Batch size must be greater than 0"
        assert self.epochs > 0, "Number of epochs must be greater than 0"
        assert self.learning_rate > 0, "Learning rate must be greater than 0"
        assert 0 <= self.dropout_rate < 1, "Dropout rate must be between 0 and 1"
        assert self.data_path, "Data path must be specified"
        assert self.annotation_path, "Annotation path must be specified"
        assert self.log_dir, "Log directory must be specified"
        assert self.checkpoint_dir, "Checkpoint directory must be specified"
    
    @classmethod
    def load_config(cls, config_path: str = os.path.join(Path(__file__).parent, "config.yaml")) -> "ModelConfig":
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
            config_path = os.path.join(Path(__file__).parent, "config.yaml") # default config path
        _config_instance = ModelConfig.load_config(config_path)
    
    return _config_instance

def reset_config():
    global _config_instance
    _config_instance = None
        