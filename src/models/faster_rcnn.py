import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import ResNet18_Weights


class FasterRCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained = True): # 2 classes: background and tumor
        super().__init__()
        
        
        backbone = resnet_fpn_backbone('resnet18', weights=ResNet18_Weights.DEFAULT)
        
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(
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
            
            num_classes=num_classes,
            )
        
        self.model.backbone = backbone # swap backbone
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets) # training, internally computes the loss
        
        return self.model(images) # inference only