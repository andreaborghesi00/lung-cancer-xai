import torch
import torch.nn as nn
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead, RetinaNet, RetinaNetHead, RetinaNet
from torchvision.models import ResNet50_Weights
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights, efficientnet_b2, EfficientNet_B2_Weights, efficientnet_b3, EfficientNet_B3_Weights
from typing import Optional, Any
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, LastLevelP6P7
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.backbone_utils import _mobilenet_extractor
from functools import partial
from utils.utils import setup_logging
from torchvision.ops import misc as misc_nn_ops

class RetinaNetResnet50(nn.Module):
    """
    RetinaNet with ResNet-50 + FPN backbone,
    customized for small/medium object detection (such as lung nodules).
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Load pretrained weights and transforms
        # self.weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        self.weights = None
        self.backbone_weights = ResNet50_Weights.DEFAULT
        self.transforms = self.backbone_weights.transforms()

        # Initialize RetinaNet
        self.model = retinanet_resnet50_fpn_v2(
            weights=self.weights,
            # weights=None,
            backbone_weights=self.backbone_weights,
            box_score_thresh=0.01,         # Confidence threshold
            box_nms_thresh=0.3,            # NMS IoU threshold
            box_detections_per_img=400,    # Max detections/image
            rpn_nms_thresh=0.7,            # RPN NMS threshold
            num_classes=num_classes
        )

        # Replace anchor generator for small scales
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((1.0,),) * len(anchor_sizes)
        self.model.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )

        # Rebuild the classification head for the new num_classes
        # Number of anchors per location (all levels have same)
        num_anchors = self.model.anchor_generator.num_anchors_per_location()[0]
        # Channels from FPN output
        # in_channels = self.model.head.classification_head.conv[0].in_channels
        in_channels = self.model.backbone.out_channels

        self.model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
        self.model.head.regression_head = RetinaNetRegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
        )
        self.model.head.regression_head._loss_type = "giou"


    def get_transform(self):
        """Return the preprocessing transforms for the model."""
        return self.transforms

    def forward(self, images, targets=None):
        """
        Forward pass.
        - During training, returns losses dict.
        - During inference, returns detections.
        """
        if self.training:
            return self.model(images, targets)
        return self.model(images)


class RetinaNetEfficientNetv2s(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.weights = EfficientNet_V2_S_Weights.DEFAULT
        self.transform = self.weights.transforms()
        self.model = self.retinanet_efficientnetv2s_fpn(
            weights=self.weights,
            progress=True,
            num_classes=num_classes,
        )
    
    def retinanet_efficientnetv2s_fpn(
    self,
    weights = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    trainable_backbone_layers: Optional[int] = 3,
    **kwargs: Any,
    ) -> RetinaNet:
        
        weights = EfficientNet_V2_S_Weights.verify(weights)
        # weights_backbone = ResNet50_Weights.verify(weights_backbone)

        backbone = efficientnet_v2_s(weights=weights, progress=progress)
        backbone_fpn = self._effientnet_backbone_fpn(backbone.features, trainable_backbone_layers=trainable_backbone_layers)
        
        anchor_generator = self._default_anchorgen()
        head = RetinaNetHead(
            backbone_fpn.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            num_classes if weights is None else len(weights.meta["categories"]),
            norm_layer=partial(nn.GroupNorm, 32),
        )
        head.regression_head._loss_type = "giou"
        
        model = RetinaNet(backbone_fpn,
                          num_classes if weights is None else len(weights.meta["categories"]), 
                          anchor_generator=anchor_generator, 
                          head=head, 
                          **kwargs)

        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
        in_channels = model.backbone.out_channels

        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
        model.head.regression_head = RetinaNetRegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
        )
        
        return model

    def _default_anchorgen(self):
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((1.0,),) * len(anchor_sizes)
        return AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        

    def _effientnet_backbone_fpn(
    self,
    backbone_features,
    trainable_backbone_layers: Optional[int] = None,
    ) -> BackboneWithFPN:
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

        # trainable_backbone_layers = 4
        if trainable_backbone_layers > num_stages:
            raise ValueError(f"trainable_layers {trainable_backbone_layers} cannot be greater than the number of stages {num_stages}")

        freeze_before = len(backbone_features) if trainable_backbone_layers == 0 else stage_indices[num_stages - trainable_backbone_layers]

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
                    extra_blocks=LastLevelP6P7(in_channels_list[-1], out_channels),
                    # norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                )
    def get_transform(self):
        """Return the preprocessing transforms for the model."""
        return self.transform

    def forward(self, images, targets=None):
        """
        Forward pass.
        - During training, returns losses dict.
        - During inference, returns detections.
        """
        if self.training:
            return self.model(images, targets)
        return self.model(images)

class RetinaNetMobileNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.weights = MobileNet_V3_Large_Weights.DEFAULT
        self.transform = self.weights.transforms()
        self.model = self.retinanet_mobilenet_fpn(
            weights=self.weights,
            progress=True,
            num_classes=num_classes,
        )
        
    def retinanet_mobilenet_fpn(self, weights, trainable_backbone_layers=6, num_classes=2, **kwargs):
        norm_layer = misc_nn_ops.FrozenBatchNorm2d 

        backbone = mobilenet_v3_large(weights=weights, progress=True, norm_layer=norm_layer)
        backbone_fpn = _mobilenet_extractor(backbone, True, trainable_backbone_layers)
        anchor_sizes = (
            (
                8,
                16,
                32,
                64,
                128,
            ),
        ) * 3
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        
        head = RetinaNetHead(
            backbone_fpn.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            num_classes if weights is None else len(weights.meta["categories"]),
            norm_layer=partial(nn.GroupNorm, 32),
        )
        head.regression_head._loss_type = "giou"
        
        model = RetinaNet(backbone_fpn,
                          num_classes if weights is None else len(weights.meta["categories"]), 
                          anchor_generator=anchor_generator, 
                          head=head, 
                          **kwargs)

        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
        in_channels = model.backbone.out_channels

        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
        model.head.regression_head = RetinaNetRegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
        )
        
        return model
    
    def get_transform(self):
        """Return the preprocessing transforms for the model."""
        return self.transform

    def forward(self, images, targets=None):
        """
        Forward pass.
        - During training, returns losses dict.
        - During inference, returns detections.
        """
        if self.training:
            return self.model(images, targets)
        return self.model(images)
