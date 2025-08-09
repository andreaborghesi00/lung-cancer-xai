import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights, efficientnet_b2, EfficientNet_B2_Weights, efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
import logging
from utils.utils import setup_logging
from config.config_2d import get_config
from typing import Optional, Any, Union, Callable, Tuple
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops
import torch.nn.functional as F
from typing import List, Dict, Optional

config = get_config()
config.validate() 

class FocalLossROIHeads(RoIHeads):
    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        ce_loss = F.cross_entropy(class_logits, labels)
        
        # Replace CE loss with Focal Loss
        # ce_loss = self.focal_loss(class_logits, labels, alpha=0.25, gamma=2.0)
        
        # Smooth L1 for regression
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        # GIoU loss
        # box_loss = self.giou_loss(box_regression[sampled_pos_inds_subset, labels_pos],
        #                           regression_targets[sampled_pos_inds_subset],
        #                           reduction="mean"
        #                           )
        
        # L2 loss
        # box_loss = F.mse_loss(
        #     box_regression[sampled_pos_inds_subset, labels_pos],
        #     regression_targets[sampled_pos_inds_subset],
        #     reduction="sum"
        # ) / labels.numel() # isn't this just a mean reduction?
                
        return ce_loss, box_loss

    def giou_loss(self, pred_boxes, target_boxes, reduction="none"):
        """
        pred_boxes, target_boxes: [N, 4] in (x1, y1, x2, y2)
        """
        # Intersection
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        # Areas
        area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_target = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

        # Union
        union = area_pred + area_target - inter
        iou = inter / (union + 1e-7)

        # Enclosing box
        x1_c = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        y1_c = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        x2_c = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        y2_c = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        area_c = (x2_c - x1_c) * (y2_c - y1_c)

        giou = iou - (area_c - union) / (area_c + 1e-7)
        loss = 1 - giou
        if reduction == 'none':
            pass
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        
        return loss

    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = alpha * (1 - pt) ** gamma * ce
        return focal.mean()
    
    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = self.fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses


class FasterRCNNResnet50(nn.Module):
    def __init__(self, num_classes=2): # 2 classes: background and tumor
        super().__init__()
        
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.transforms = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
        
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights,
            # RPN parameters
            box_score_thresh=0.01,  # Lower confidence threshold (default is usually 0.05)
            box_nms_thresh=0.3,     # NMS threshold
            box_detections_per_img=400, # Increased max detections per image
            rpn_nms_thresh=0.7, 
            )
        
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def get_transform(self):
        return self.transforms
    
    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets) # training, internally computes the loss
        
        return self.model(images) # inference only
    
class FasterRCNNMobileNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        
        self.weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.transforms = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.transforms()
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=self.weights,
            box_score_thresh=0.01,  # Lower confidence threshold (default is usually 0.05)
            box_nms_thresh=0.3,     # NMS threshold
            box_detections_per_img=400, # Increased max detections per image
            rpn_nms_thresh=0.7,     # NMS threshold for RPN proposals
            trainable_backbone_layers=5
            
        ) 
        
        # print model backbone architecture
        self.logger.debug(self.model.backbone)
        
        # change the number of expected classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    def get_transform(self):
        return self.transforms
    
    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        
        return self.model(images)


class FasterRCNNEfficientNetv2s(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        
        self.weights = EfficientNet_V2_S_Weights.DEFAULT
        self.transforms = EfficientNet_V2_S_Weights.DEFAULT.transforms()
        
        # Load the EfficientNet backbone

        # Create the Faster R-CNN model with the EfficientNet backbone
        self.model = self.fasterrcnn_efficientnet_v2s_fpn(True, num_classes=num_classes, weights_backbone=self.weights)
        
    def inflate_efficientnet_input_channels(self, model: nn.Module, in_channels: int):
        """Inflate EfficientNet first conv layer weights to handle more channels."""
        conv = model.features[0][0]  # first Conv2d layer in EfficientNetV2-S
        assert isinstance(conv, nn.Conv2d)
        old_weights = conv.weight.data  # shape: [out_channels, 3, k, k]

        if in_channels == 3:
            return model  # nothing to do

        # Create new weights
        new_weights = torch.zeros((conv.out_channels, in_channels, *conv.kernel_size))

        # Copy existing RGB weights
        new_weights[:, :3, :, :] = old_weights

        # Initialize extra channels
        if in_channels > 3:
            extra = in_channels - 3
            # Average pretrained weights over RGB channels
            mean_weight = old_weights.mean(dim=1, keepdim=True)  # shape: [out_channels, 1, k, k]
            # Repeat averaged weights for new channels
            new_weights[:, 3:, :, :] = mean_weight.repeat(1, extra, 1, 1)

        # Replace conv layer
        conv_new = nn.Conv2d(in_channels, conv.out_channels,
                            kernel_size=conv.kernel_size,
                            stride=conv.stride,
                            padding=conv.padding,
                            bias=(conv.bias is not None))
        conv_new.weight = nn.Parameter(new_weights)
        if conv.bias is not None:
            conv_new.bias.data = conv.bias.data

        # Swap into model
        model.features[0][0] = conv_new
        return model


    def _effientnet_backbone_fpn(
        self,
        backbone_features,
        
    ) -> FasterRCNN:
        current_channels = []
        stage_indices = []
        for i, block in enumerate(backbone_features):
            if hasattr(block, 'out_channels'):
                # print(f"Block {i}: {block.out_channels}")
                current_channels.append(block.out_channels)
                stage_indices.append(i)
            elif hasattr(block[-1], 'out_channels'):
                # print(f"Block {i} (last inner): {block[-1].out_channels}")
                current_channels.append(block[-1].out_channels)
                stage_indices.append(i)
        num_stages = len(stage_indices)
        # returned_layers = [num_stages - 4, num_stages - 3, num_stages - 2, num_stages - 1]
        # returned_layers = {f"{i}": str(i) for i in returned_layers}

        returned_layers = {
            "2": "0",
            "4": "1",
            "6": "2",
            "7": "3",
        }

        trainable_layers = 4
        if trainable_layers > num_stages:
            raise ValueError(f"trainable_layers {trainable_layers} cannot be greater than the number of stages {num_stages}")

        freeze_before = len(backbone_features) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

        for b in backbone_features[:freeze_before]:
            for parameter in b.parameters():
                parameter.requires_grad_(False)

        in_channels_list = [current_channels[i] for i in [2,4,6,7]]
        out_channels = 256


        return BackboneWithFPN(
                    backbone=backbone_features,
                    return_layers=returned_layers,
                    in_channels_list=in_channels_list,
                    out_channels=out_channels,
                    extra_blocks=LastLevelMaxPool(),
                    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                )

    def fasterrcnn_efficientnet_v2s_fpn(self,
    # weights: Optional[EfficientNet_V2_S_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = 2,
    weights_backbone: Optional[EfficientNet_V2_S_Weights] = EfficientNet_V2_S_Weights.DEFAULT,
    # trainable_backbone_layers: Optional[int] = None,
    # **kwargs: Any,
    ) -> FasterRCNN:

        backbone = efficientnet_v2_s(weights=weights_backbone, progress=progress)
        # backbone = self.inflate_efficientnet_input_channels(backbone, in_channels=3)  # Ensure input channels are 3
        backbone_fpn = self._effientnet_backbone_fpn(backbone.features)

        anchor_sizes = tuple([(4,), (8,), (16,), (32,), (64,),]) 
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        
        model = FasterRCNN(
            backbone_fpn,
            num_classes if weights_backbone is None else len(weights_backbone.meta['categories']),
            rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
            rpn_score_thresh=0.05,
            box_score_thresh=0.01,  # Lower confidence threshold (default is usually 0.05)
            box_nms_thresh=0.3,     # NMS threshold
            box_detections_per_img=400, # Increased max detections per image
            rpn_nms_thresh=0.7,  
        )
        
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(backbone_fpn.out_channels * resolution**2, representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, num_classes)

        model.roi_heads = FocalLossROIHeads(
                box_roi_pool=box_roi_pool,
                box_head=box_head,
                box_predictor=box_predictor,
                fg_iou_thresh=0.5,
                bg_iou_thresh=0.5,
                batch_size_per_image=512,
                positive_fraction=0.25,
                bbox_reg_weights=None,
                score_thresh=0.05,
                nms_thresh=0.5,
                detections_per_img=100
        )
        
        model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],  # Use the exact same names as in return_layers
            output_size=7,
            sampling_ratio=2
        )
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
        
        return model
    
    
    def get_transform(self):
        return self.transforms
    
    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        
        return self.model(images)
    
class FasterRCNNEfficientNetB2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        
        self.weights = EfficientNet_B2_Weights.DEFAULT
        self.transforms = EfficientNet_B2_Weights.DEFAULT.transforms()
        
        # Load the EfficientNet backbone

        # Create the Faster R-CNN model with the EfficientNet backbone
        self.model = self.fasterrcnn_efficientnet_b2_fpn(True, num_classes=num_classes, weights_backbone=self.weights)
        

    def _effientnet_backbone_fpn(
        self,
        backbone_features,
        
    ) -> FasterRCNN:
        current_channels = []
        stage_indices = []
        for i, block in enumerate(backbone_features):
            if hasattr(block, 'out_channels'):
                # print(f"Block {i}: {block.out_channels}")
                current_channels.append(block.out_channels)
                stage_indices.append(i)
            elif hasattr(block[-1], 'out_channels'):
                # print(f"Block {i} (last inner): {block[-1].out_channels}")
                current_channels.append(block[-1].out_channels)
                stage_indices.append(i)
        num_stages = len(stage_indices)
        # returned_layers = [num_stages - 4, num_stages - 3, num_stages - 2, num_stages - 1]
        # returned_layers = {f"{i}": str(i) for i in returned_layers}

        returned_layers = {
            "2": "0",
            "4": "1",
            "6": "2",
            "8": "3",
        }

        trainable_layers = 4
        if trainable_layers > num_stages:
            raise ValueError(f"trainable_layers {trainable_layers} cannot be greater than the number of stages {num_stages}")

        freeze_before = len(backbone_features) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

        for b in backbone_features[:freeze_before]:
            for parameter in b.parameters():
                parameter.requires_grad_(False)

        in_channels_list = [
            backbone_features[2][-1].out_channels,
            backbone_features[4][-1].out_channels,
            backbone_features[6][-1].out_channels,
            backbone_features[8].out_channels
        ]
        
        out_channels = 256

        return BackboneWithFPN(
                    backbone=backbone_features,
                    return_layers=returned_layers,
                    in_channels_list=in_channels_list,
                    out_channels=out_channels,
                    extra_blocks=LastLevelMaxPool(),
                    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                )

    def fasterrcnn_efficientnet_b2_fpn(self,
    # weights: Optional[EfficientNet_V2_S_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = 2,
    weights_backbone: Optional[EfficientNet_V2_S_Weights] = EfficientNet_V2_S_Weights.DEFAULT,
    # trainable_backbone_layers: Optional[int] = None,
    # **kwargs: Any,
    ) -> FasterRCNN:

        backbone = efficientnet_b2(weights=weights_backbone, progress=progress)
        backbone_fpn = self._effientnet_backbone_fpn(backbone.features)

        # anchor_sizes = tuple([(8,), (16,), (32,), (64,), (128,)]) 
        anchor_sizes = tuple([(16,), (32,), (64,), (128,), (256,)]) 
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        
        model = FasterRCNN(
            backbone_fpn,
            num_classes if weights_backbone is None else len(weights_backbone.meta['categories']),
            rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
            rpn_score_thresh=0.05,
            box_score_thresh=0.01,  # Lower confidence threshold (default is usually 0.05)
            box_nms_thresh=0.3,     # NMS threshold
            box_detections_per_img=400, # Increased max detections per image
            rpn_nms_thresh=0.7,  
        )
        
        model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
            # featmap_names=['4', '5', '6', '7'],  # Use the exact same names as in return_layers
            # featmap_names=['2', '4', '6', '7'],  # Use the exact same names as in return_layers
            featmap_names=['0', '1', '2', '3'],  # Use the exact same names as in return_layers
            output_size=7,
            sampling_ratio=2
        )
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
        
        return model
    
    
    def get_transform(self):
        return self.transforms
    
    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        
        return self.model(images)