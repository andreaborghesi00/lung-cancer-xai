from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.base_cam import BaseCAM
from typing import Dict, Any, List, Union
import torch
import torch.nn as nn
import numpy as np
import logging
from utils.utils import setup_logging
from typing import Callable, List, Optional, Tuple
from config.config_2d import get_config
from tqdm import tqdm
import ttach as tta
import torchvision
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
import torch.nn.functional as F

def fasterrcnn_reshape_transform(x):
    target_size = x['pool'].size()[-2 : ]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations

config = get_config()
config.validate()

class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, scores, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.scores = scores
        self.iou_threshold = iou_threshold
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    def __call__(self, model_outputs):
        output = torch.tensor([0.0], device=self.device)
        
        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label, score in zip(self.bounding_boxes, self.labels, self.scores):
            box = torch.Tensor(box[None, :]).to(self.device)

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                # score = ious[0, index] + model_outputs["scores"][index]
                score = ious[0, index] * (1 - (model_outputs["scores"][index] - score)**2)
                output = output + score
        return output


class CustomGradCAM():
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        tta_transforms: Optional[tta.Compose] = None,
    ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers

        # Use the same device as the model.
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        if tta_transforms is None:
            self.tta_transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
        else:
            self.tta_transforms = tta_transforms

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            activation_tensor = activation_tensor.to(self.device)

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

            input_tensors = input_tensor[:, None,
                                         :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in range(0, tensor.size(0), BATCH_SIZE):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        if len(input_tensor.shape) == 4:
            width, height = input_tensor.size(-1), input_tensor.size(-2)
            return width, height
        elif len(input_tensor.shape) == 5:
            depth, width, height = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
            return depth, width, height
        else:
            raise ValueError("Invalid input_tensor shape. Only 2D or 3D images are supported.")

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> np.ndarray:
        activations_list = np.array([a.cpu().data.numpy() for a in self.activations_and_grads.activations])
        grads_list = np.array([g.cpu().data.numpy() for g in self.activations_and_grads.gradients])
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        cams = []
        for transform in self.tta_transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor, targets, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor, targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

class SSCAM(BaseCAM):
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        kernel_size: int = 5,
        sigma: float = 2.0,
        num_samples: int = 35,
    ):
        super(SSCAM, self).__init__(
            model, target_layers, reshape_transform=reshape_transform, uses_gradients=False
        )
        self.num_samples = num_samples
        self.sigma = sigma
        self._distrib = torch.distributions.normal.Normal(0.0, self.sigma)
    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        with torch.no_grad():
            # Convert activations to tensor and move to device
            activation_tensor = torch.from_numpy(activations).to(self.device)   # B, C, H, W (1, 256, 25, 25)         

            # Upsample smoothed activations to input size
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[-2:])
            upsampled = upsample(activation_tensor) # upsampled to (B, C, H_in, W_in)

            # Normalize to [0, 1] per activation map
            maxs = upsampled.view(upsampled.size(0), upsampled.size(1), -1).max(dim=-1)[0] # max per channel of activation map
            mins = upsampled.view(upsampled.size(0), upsampled.size(1), -1).min(dim=-1)[0] # min per channel of activation map
            maxs = maxs[:, :, None, None] # add h w dimension
            mins = mins[:, :, None, None] # add h w dimension
            upsampled_normalized = (upsampled - mins) / (maxs - mins + 1e-8)

            # Mask input with normalized activations and compute scores
            input_tensors = input_tensor[:, None, :, :] * upsampled_normalized[:, :, None, :, :]

            # baseline_outputs = self.model(input_tensor)
            # baseline_target = targets[0]
            # baseline_logits = baseline_target(baseline_outputs).cpu().numpy()
            
            
            # Process in batches to reduce memory usage
            BATCH_SIZE = 32
            scores = np.zeros((activations.shape[0], activations.shape[1]), dtype=np.float32)
            idx = 0
            for target, act in zip(targets, upsampled_normalized):
                # Add noise
                for _ in tqdm(range(self.num_samples)):
                    noise = self._distrib.sample(act.size()).to(self.device)
                    noise.div_(20.) # scale down the noise, otherwise it will overpower the activation map
                    scored_input = (act + noise)[:, None, :, :] * input_tensor # type 1
                    # scored_input = (act[:, None, :, :] * input_tensor) + noise [:, None,:, :,] # type 2
                    curr_score = []
                    for i in range(0, scored_input.size(0), BATCH_SIZE):
                        batch = scored_input[i:i + BATCH_SIZE, :]
                        outputs = [target(o).cpu().item() for o in self.model(batch)]
                        curr_score.extend(outputs)
                    scores[idx] += np.array(curr_score)
                    

            scores = torch.Tensor(scores)
            scores = (scores.div_(self.num_samples))
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights

class CAMExplainer:
    def __init__(self, model: nn.Module, target_layer: Any, cam_class = EigenCAM, reshape_transform: Callable = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        
        self.model = model
        self.target_layer = target_layer
        if cam_class.__name__ == "AblationCAM":
            self.logger.info("Using AblationCAM")
            self.explainer = cam_class(model, target_layers=target_layer, reshape_transform=reshape_transform, ablation_layer=AblationLayerFasterRCNN(), ratio_channels_to_ablate=1.0)
        else:
            self.explainer = cam_class(model, target_layers=target_layer, reshape_transform=reshape_transform)
        
    def explain(self, image: torch.Tensor, labels: Union[int, List[int]], bboxes: List[Dict[str, int]], scores, iou_threshold: int = 0.5) -> Any:
        # self.model.eval()
        # Debug prints
        self.logger.debug(f"Input image requires_grad: {image.requires_grad}")
        self.logger.debug(f"Input image device: {image.device}")
        self.logger.debug(f"Model device: {next(self.model.parameters()).device}")
        
        # Ensure image has gradients
        if not image.requires_grad:
            image = image.detach().clone()
            image.requires_grad = True
        
        if isinstance(labels, int):
            labels = [labels]

        targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=bboxes, scores=scores , iou_threshold=iou_threshold)]
        cam = self.explainer(input_tensor=image, targets=targets)
        return cam

    def visualize(self, image: torch.Tensor, cam: torch.Tensor) -> np.ndarray:
        visualization = show_cam_on_image(image, cam, use_rgb=True)
        return visualization