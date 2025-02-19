from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.base_cam import BaseCAM
from typing import Dict, Any, List, Union
import torch
import torch.nn as nn
import numpy as np

class CAMExplainer:
    def __init__(self, model: nn.Module, target_layer: Any, target_class: int, cam_class: BaseCAM = GradCAM):
        self.model = model
        self.target_layer = target_layer
        self.target_class = target_class
        self.explainer = cam_class(model, target_layers=target_layer)
        
    def explain(self, image: torch.Tensor, labels: Union[int, List[int]], bboxes: List[Dict[str, int]], iou_threshold: int = 0.5) -> Any:
        self.model.eval()
        if isinstance(labels, int):
            labels = [labels]

        targets = [FasterRCNNBoxScoreTarget(labels, bboxes, iou_threshold=iou_threshold)]
        cam = self.explainer(input_tensor=image, targets=targets)
        return cam

    def visualize(self, image: torch.Tensor, cam: torch.Tensor) -> np.ndarray:
        visualization = show_cam_on_image(image, cam, use_rgb=True)
        return visualization
