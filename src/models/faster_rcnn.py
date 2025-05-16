import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
import logging
from utils.utils import setup_logging
from config.config_2d import get_config
from typing import Optional, Any, Union, Callable, Tuple
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops

config = get_config()
config.validate() 

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


class FasterRCNNEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        
        self.weights = EfficientNet_V2_S_Weights.DEFAULT
        self.transforms = EfficientNet_V2_S_Weights.DEFAULT.transforms()
        
        # Load the EfficientNet backbone

        # Create the Faster R-CNN model with the EfficientNet backbone
        self.model = self.fasterrcnn_efficientnet_v2s_fpn(True, num_classes=num_classes, weights_backbone=self.weights)
        

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
            "2": "2",
            "4": "4",
            "6": "6",
            "7": "7",
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
        backbone_fpn = self._effientnet_backbone_fpn(backbone.features)

        anchor_sizes = tuple([(8,), (16,), (32,), (64,), (128,),]) 
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