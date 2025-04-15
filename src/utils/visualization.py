import matplotlib.pyplot as plt
import torch
from typing import Tuple, Union
from PIL import Image
import numpy as np
from config.config_2d import get_config
from pathlib import Path
import utils.utils as utils
import logging


class Visualizer:
    def __init__(self, model_name: str = None):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        utils.setup_logging()
        
        self.visualization_dir = Path(self.config.visualization_dir)
        if model_name is not None:
            self.visualization_dir = self.visualization_dir / model_name
        self.visualization_dir = self.visualization_dir / self.config.visualization_experiment_name
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
          
          
    def _add_bbox_to_plot(self, ax, box, color, label=None, score=None):
        """
        Helper function to add a bounding box rectangle and optional score text to a plot axes.

        Args:
            ax: Matplotlib axes object to add the rectangle to.
            box: Bounding box coordinates (x1, y1, x2, y2).
            color: Color of the bounding box and text.
            label: Label for the bounding box (for legend, only applied to the first box of each type).
            score: Confidence score for the predicted box (optional, for predicted boxes).
        """
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1),  # bottom left corner
            x2 - x1,   # width
            y2 - y1,   # height
            fill=False,
            color=color,
            linewidth=2,
            label=label
        )
        ax.add_patch(rect)
        if score is not None:
            ax.text(
                x1, y1 - 5,
                f'{label.split()[0]}: {score:.2f}', # Use label prefix for text
                color=color,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8)
            )

    
    def display_tomo_bboxes(
        self,
        tomo: Union[torch.Tensor, np.ndarray], 
        boxes: Union[torch.Tensor, np.ndarray],
        true_boxes: Union[torch.Tensor, np.ndarray],
        scores: Union[torch.Tensor, np.ndarray],
        skips: np.ndarray,
        save_dir: str = None,
        figsize: Tuple[int, int] = (12, 8)
        ) -> None:
        """
        Visualize bounding boxes on a tomography
        
        Args:
            tomo: Tomography data as tensor or numpy array of shape (D, C, H, W)
            boxes: Predicted boxes (N, 4) in (x1, y1, x2, y2) format
            true_boxes: Ground truth boxes (M, 4) in (x1, y1, x2, y2) format
            scores: Confidence scores for predicted boxes
            save_dir: Directory to save images (will be created if it doesn't exist)
            figsize: Figure size for the plot
        """
        tomo = utils.to_numpy(tomo if isinstance(tomo, torch.Tensor) else tomo)
        
        # Determine number of slices in the tomography
        num_slices = tomo.shape[0] if len(tomo.shape) == 4 else 1
        
        # Create save directory if provided
        if save_dir is not None:
            save_path = self.visualization_dir / save_dir
            save_path.mkdir(parents=True, exist_ok=True)
        
        for slice_idx in range(num_slices):
            # Extract current slice
            current_slice = tomo[slice_idx] 
            curr_boxes = boxes[slice_idx]
            curr_true_boxes = true_boxes[slice_idx]
            curr_scores = scores[slice_idx]
            curr_skip = skips[slice_idx]
            # Create filename if saving
            filename = Path(f"slice_{slice_idx}.png") if save_dir is not None else None
            
            # Use the display_bboxes method to show/save the visualization
            self.display_bboxes(
                input=current_slice,
                pred_boxes=curr_boxes,
                true_boxes=curr_true_boxes,
                scores=curr_scores,
                skip=curr_skip,
                filename=save_dir / filename if save_dir is not None and filename is not None else None,
                figsize=figsize
            )

    
    
    def display_bboxes(
        self,
        input: Union[str, np.ndarray],
        pred_boxes: Union[torch.Tensor, np.ndarray],
        true_boxes: Union[torch.Tensor, np.ndarray],
        scores: torch.Tensor,
        filename: str = None,
        figsize: Tuple[int, int] = (12, 8),
        skip: bool = False
    ) -> None:
        """
        Visualize predicted and ground truth boxes on the same image
        Args:
            image_path: Path to the image or the image as a numpy array
            pred_boxes: Predicted boxes (N, 4) in (x1, y1, x2, y2) format
            true_boxes: Ground truth boxes (M, 4) in (x1, y1, x2, y2) format
            scores: Confidence scores for predicted boxes
            filename: name of the visualization, if present the image will be saved on the configured visualization path
        """
        # Load and convert image
        if isinstance(input, str):
            image = Image.open(input)
            image_array = np.array(image)
        elif isinstance(input, np.ndarray):
            image_array = input
        else:
            raise ValueError("Input must be a path to an image or a numpy array")
        
        image_array = image_array[0] if len(image_array.shape) == 3 else image_array
        pred_boxes = utils.to_numpy(pred_boxes)
        true_boxes = utils.to_numpy(true_boxes)
        scores = utils.to_numpy(scores)
        
        plt.figure(figsize=figsize)
        plt.imshow(image_array, cmap='gray')
        ax = plt.gca() # get current axes
        
        # Plot predicted boxes in red
        # for i, box_score in enumerate(zip(pred_boxes, scores if scores is not None else [None]*len(pred_boxes))):
        scores = scores if scores is not None else [None]*len(pred_boxes)
        for box, score in zip(pred_boxes, scores):
            # box, score = box_score
            label = 'Predicted'
            self._add_bbox_to_plot(ax, box, 'red' if skip == False else "purple", label, score)
        
        # Plot ground truth boxes in green
        for box in true_boxes:
            label = 'Ground truth'
            self._add_bbox_to_plot(ax, box, 'green', label)
        
        plt.legend()
        plt.axis('off')
        
        if filename is not None:
            plt.savefig(self.visualization_dir / filename, bbox_inches='tight', pad_inches=0)
        else: 
            plt.show()
        plt.close()