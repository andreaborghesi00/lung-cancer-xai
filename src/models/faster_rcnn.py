import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import logging
from utils.utils import setup_logging
class FasterRCNNResnet50(nn.Module):
    def __init__(self, num_classes=2): # 2 classes: background and tumor
        super().__init__()
        
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.transforms = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
        
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights,
            # RPN parameters
            rpn_pre_nms_top_n_train=2000,  # Number of proposals to keep before NMS (during training)
            rpn_pre_nms_top_n_test=1000,   # Number of proposals to keep before NMS (during testing)
            rpn_post_nms_top_n_train=2000, # Number of proposals to keep after NMS (during training)
            rpn_post_nms_top_n_test=1000,  # Number of proposals to keep after NMS (during testing)
            rpn_nms_thresh=0.7,            # NMS threshold used for RPN proposals
            rpn_fg_iou_thresh=0.5,         # IoU threshold for positive anchors
            rpn_bg_iou_thresh=0.3,         # IoU threshold for negative anchors
            rpn_batch_size_per_image=256,  # Number of anchors per image to sample for RPN training
            rpn_positive_fraction=0.5,     # Proportion of positive anchors for RPN training
            
            # ROI parameters
            box_score_thresh=0.05,         # Minimum score threshold (assuming scores are probabilities)
            box_nms_thresh=0.5,            # NMS threshold for prediction
            box_detections_per_img=256,    # Maximum number of detections per image
            box_fg_iou_thresh=0.5,         # IoU threshold for positive boxes
            box_bg_iou_thresh=0.3,         # IoU threshold for negative boxes
            box_batch_size_per_image=512,  # Number of ROIs per image to sample for training
            box_positive_fraction=0.25,    # Proportion of positive ROIs for training
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
            
        )  # as of now it expects 91 classes (COCO dataset)
        
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
            