import numpy as np
import logging
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Union
from PIL import Image
from config.config import get_config
from pathlib import Path


def roi_overlay(image, roi, color=(255, 0, 0), lw=2):
    x, y, w, h = roi
    image = image.copy()
    image[y:y+h, x:x+lw] = color
    image[y:y+h, x+w:x+w+lw] = color
    image[y:y+lw, x:x+w] = color
    image[y+h:y+h+lw, x:x+w] = color
    return image

def greyscale_to_rgb(image):
    return np.repeat(image[..., np.newaxis], 3, axis=-1)

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(model_path: str, model_class: torch.nn.Module, device="cuda" if torch.cuda.is_available() else "cpu") -> torch.nn.Module:
    """Load the model from checkpoint"""   
    model = model_class().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location="cuda")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def to_numpy(data: Union[torch.Tensor, np.ndarray, None]) -> Union[np.ndarray, None]:
    """
    Converts input data to a NumPy array if it's a torch.Tensor.
    Returns the input as is if it's already a NumPy array or None.
    Raises TypeError for invalid input types.

    Args:
        data: Input data, can be a torch.Tensor, np.ndarray, or None (e.g., for optional scores).

    Returns:
        NumPy array representation of the data, or None if input was None.

    Raises:
        TypeError: If the input is not a torch.Tensor, np.ndarray, or None.
    """
    if data is None:
        return None
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Input data must be a torch.Tensor, np.ndarray, or None, but got {type(data)}")