import numpy as np
import logging
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Union
from PIL import Image
from pathlib import Path
from typing import Optional

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
    
def visualize_sample_slice(image: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray, epoch: int, gt_boxes: np.ndarray = None, save_dir: Optional[str] = None):
    depth = image.shape[-1]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0, :, :, depth//2], cmap="gray") # show the middle slice
    plt.axis('off')
    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, z1, x2, y2, z2 = box
        w = x2 - x1
        h = y2 - y1
        plt.gca().add_patch(plt.Rectangle((y1, x1), w, h, linewidth=1.5, edgecolor='r', facecolor='none'))
        plt.text(y1, x1-10, str((score)), color='r')

    if gt_boxes is not None:
        for gt_box in gt_boxes:
            x1, y1, z1, x2, y2, z2 = gt_box
            w = x2 - x1
            h = y2 - y1
            plt.gca().add_patch(plt.Rectangle((y1, x1), w, h, linewidth=1.5, edgecolor='g', facecolor='none'))
        
    if save_dir is not None:
        plt.savefig(Path(save_dir) / f"sample_{epoch}_slice_{depth//2}.png")
    else:
        plt.savefig(f"sample_{epoch}_slice_{depth//2}.png")