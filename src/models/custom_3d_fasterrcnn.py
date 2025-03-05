from typing import Dict, List, Optional, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN 
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, GeneralizedRCNN, FastRCNNPredictor, TwoMLPHead
from torchvision.ops import MultiScaleRoIAlign, FeaturePyramidNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class AdaptiveVolumeBackbone(nn.Module):
    """
    Custom 3D backbone for Faster RCNN
    Handles volumetric data with fixed H,W but variable D
    Returns 2D feature maps for use in FPN after adaptive pooling
    """
    def __init__(self, in_channels=1, pretrained=False):
        super(AdaptiveVolumeBackbone, self).__init__()
        self.in_channels_list = [64, 128, 256, 512, 512]
        self.out_channels = self.in_channels_list[-1]
        
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, self.in_channels_list[0], kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
            nn.BatchNorm3d(self.in_channels_list[0]),
            nn.SiLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # NOTE: hardcoding the number of blocks for now
        self.layer2 = self._make_layer(self.in_channels_list[0], self.in_channels_list[1], 2)
        self.layer3 = self._make_layer(self.in_channels_list[1], self.in_channels_list[2], 4, stride=(1, 2, 2))
        self.layer4 = self._make_layer(self.in_channels_list[2], self.in_channels_list[3], 4, stride=(1, 2, 2))
        self.layer5 = self._make_layer(self.in_channels_list[3], self.in_channels_list[4], 2, stride=(1, 2, 2))
        
        # adaptive Pooling to handle variable depth
        # maps to fixed depth size
        self.adaptive_pool_d = nn.AdaptiveAvgPool3d((16, None, None))
        
        # initialize weights, kaiming normal, batchnorm to (1, 0)
        if not pretrained:
            self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=(1, 1, 1)):
        layers = []
        layers.append(self._residual_block(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels, out_channels, stride=(1, 1, 1)):
        downsample = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
        return ResidualBlock3D(in_channels, out_channels, stride, downsample)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    
    def forward(self, x):
        # Apply layers
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        
        # Adaptive pooling to get consistent depth dimension
        c1 = self.adaptive_pool_d(c1)
        c2 = self.adaptive_pool_d(c2)
        c3 = self.adaptive_pool_d(c3)
        c4 = self.adaptive_pool_d(c4)
        c5 = self.adaptive_pool_d(c5)
        
        # Convert 3D feature maps to 2D by averaging across depth dimension
        # This preserves spatial information while making it compatible with FasterRCNN
        # B x C x D x H x W -> B x C x H x W
        c1_2d = torch.mean(c1, dim=2)
        c2_2d = torch.mean(c2, dim=2)
        c3_2d = torch.mean(c3, dim=2)
        c4_2d = torch.mean(c4, dim=2)
        c5_2d = torch.mean(c5, dim=2)
        
        return {
            'feat0': c1_2d,
            'feat1': c2_2d,
            'feat2': c3_2d,
            'feat3': c4_2d,
            'feat4': c5_2d
        }


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), downsample=None):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.silu(out)
        
        return out


def create_3d_faster_rcnn(num_classes: int = 2, 
                          pretrained_backbone: bool = False, 
                          in_channels: int = 1, 
                          **kwargs: Any):
    """
    Creates a Faster R-CNN model with a 3D custom backbone
    The backbone processes 3D volumes but outputs 2D feature maps compatible with FasterRCNN
    
    Args:
        num_classes: Number of classes to detect (including background)
        pretrained_backbone: Whether to use pretrained weights for the backbone
        in_channels: Number of input channels (typically 1 for CT/MRI)
    """
    # Create a custom backbone
    backbone = AdaptiveVolumeBackbone(in_channels=in_channels, pretrained=pretrained_backbone)
    backbone = _3dbackbone_fpn_exctractor(backbone, norm_layer=nn.BatchNorm2d) # perhaps also a norm layer? idk resnet is also doing it
    rpn_anchor_generator = _default_anchor_gen()
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2) # 2 conv layers, ripped from source of fasterrccn_resnet50_fpn_v2
    box_head = FastRCNNConvFCHead((backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d) # 7x7 is the output size of the ROI Pooler
    
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        **kwargs
    )
    
    return model

def _default_anchor_gen():
        # Define anchor sizes and aspect ratios
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    # RPN anchor generator
    rpn_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    return rpn_anchor_generator

def _3dbackbone_fpn_exctractor(backbone: AdaptiveVolumeBackbone, 
                               norm_layer: Optional[Callable[..., nn.Module]] = None
                               ) -> BackboneWithFPN:
    """
    Adds an FPN on top of the Adaptive Volume Backbone
    """
    return BackboneWithFPN(backbone, backbone.in_channels_list, backbone.out_channels, norm_layer=norm_layer)
    

class BackboneWithFPN(nn.Module):
    """
    Wrapper to combine the backbone with FPN
    """
    def __init__(self, 
                 backbone: nn.Module,
                 in_channels_list: List[int],
                 out_channels: int,
                 norm_layer: Optional[Callable[..., nn.Module]]=None
                 ) -> None:
        super(BackboneWithFPN, self).__init__()
        
        
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            norm_layer=norm_layer
        )
        self.out_channels = out_channels
        
    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x) # FPN expects an OrderedDict of feature maps, which is what the backbone is returning
        return x
    
class VolumetricDataLoader:
    """
    Helper for loading and processing volumetric data with target bboxes
    """
    @staticmethod
    def preprocess_volume(volume):
        """
        Preprocesses a 3D volume for input to the model
        Assumes volume is a numpy array with shape [D, H, W]
        
        Returns:
            torch tensor with shape [1, D, H, W]
        """
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()
        
        # Add channel dimension if not present
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        
        return volume
    
    @staticmethod
    def convert_single_bbox_to_output(bbox, volume_shape):
        """
        Converts a 3D bounding box (x1,y1,z1,x2,y2,z2) to a 2D bounding box (x1,y1,x2,y2)
        by taking the maximum extent in the z-direction
        
        Args:
            bbox: Tensor of shape [6] with format [x1,y1,z1,x2,y2,z2]
            volume_shape: The shape of the volume [D,H,W]
            
        Returns:
            Tensor of shape [4] with format [x1,y1,x2,y2]
        """
        x1, y1, z1, x2, y2, z2 = bbox
        
        # Create 2D bbox that encompasses the full 3D bbox projection
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

class Faster3DRCNN(GeneralizedRCNN):
    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchor_gen()
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["feat0", "feat1", "feat2", "feat3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size) # should it be out_channels * resolution**3 for 3D?

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        super().__init__(backbone, rpn, roi_heads, transform)
