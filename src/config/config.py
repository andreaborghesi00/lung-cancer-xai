from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    # Model architecture
    image_input_channels: int = 1
    num_classes: int = 2
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    dropout_rate: float = 0.3

    # Paths and Logging
    data_path: str = "../../data/slices_segmentate_png"
    log_dir: str = "../../logs"
    checkpoint_dir: str = "../../checkpoints"