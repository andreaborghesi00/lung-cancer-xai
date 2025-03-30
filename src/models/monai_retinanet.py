import logging
import torch
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.networks.nets import resnet
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)

logger = logging.getLogger(__name__)

def _create_anchor_generator(returned_layers, base_anchor_shapes):
    """Create an anchor generator with specified shapes and feature map scales."""
    return AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(returned_layers) + 1)], 
        base_anchor_shapes=base_anchor_shapes,
    )

def _build_backbone(n_input_channels, conv1_t_stride):
    """Build ResNet backbone network."""
    conv1_t_size = [max(7, 2 * s + 1) for s in conv1_t_stride]  # kernel size must be odd
    return resnet.ResNet(
        block=resnet.ResNetBottleneck,
        layers=[3, 4, 6, 3],
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=n_input_channels,
        conv1_t_stride=conv1_t_stride,
        conv1_t_size=conv1_t_size,
    )

def _build_feature_extractor(backbone, spatial_dims, returned_layers):
    """Build feature extractor with FPN."""
    return resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=spatial_dims,
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        returned_layers=returned_layers,
    )

def _build_network(feature_extractor, spatial_dims, num_classes, num_anchors, returned_layers):
    """Build RetinaNet network."""
    size_divisible = [s * 2 * 2 ** max(returned_layers) for s in feature_extractor.body.conv1.stride]
    return torch.jit.script(
        RetinaNet(
            spatial_dims=spatial_dims,
            num_classes=num_classes,
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
    )

def _configure_detector(detector,
                        score_thresh,
                        nms_thresh,
                        sw_roi_size=[512, 512, 320],
                        sw_overlap=0.10,
                        sw_batch_size=1,
                        ):
    """Configure detector for training and validation."""
    # Set training components
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler( 
        batch_size_per_image=64,
        positive_fraction=0.3,
        pool_size=20,
        min_neg=16,
    )
    detector.set_target_keys(box_key="box", label_key="label")

    # Set validation components
    detector.set_box_selector_parameters(
        score_thresh=score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=nms_thresh,
        detections_per_img=100,
    )
    
    detector.set_sliding_window_inferer(
        roi_size=sw_roi_size,
        overlap=sw_overlap,
        sw_batch_size=sw_batch_size,
        mode="constant",
        device="cpu",
    ) 
    
    return detector

def _create_retinanet_from_scratch(
    n_input_channels=1,
    num_classes=2,
    spatial_dims=3,
    returned_layers=[1, 2],
    base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
    conv1_t_stride=[2, 2, 1]
):
    """Create a new RetinaNet model from scratch."""
    # Build anchor generator
    anchor_generator = _create_anchor_generator(returned_layers, base_anchor_shapes)
    
    # Build network components
    backbone = _build_backbone(n_input_channels, conv1_t_stride)
    feature_extractor = _build_feature_extractor(backbone, spatial_dims, returned_layers)
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    net = _build_network(feature_extractor, spatial_dims, num_classes, num_anchors, returned_layers)
    
    return net, anchor_generator

def load_retinanet_from_checkpoint(checkpoint_path):
    """Load a RetinaNet model from a checkpoint."""
    logger.info(f"Loading pretrained model from {checkpoint_path}")
    return torch.jit.load(checkpoint_path)

def create_retinanet_detector(
    device,
    pretrained=False,
    pretrained_path=None,
    n_input_channels=1,
    num_classes=2,
    spatial_dims=3,
    returned_layers=[1, 2],
    base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
    conv1_t_stride=[2, 2, 1],
    score_thresh=0.02,
    nms_thresh=0.22,
    debug=False
):
    """Create and configure a RetinaNet detector."""
    # Build anchor generator
    anchor_generator = _create_anchor_generator(returned_layers, base_anchor_shapes)
    
    # Get network based on whether we're using a pretrained model or not
    if not pretrained:
        net, anchor_generator = _create_retinanet_from_scratch(
            n_input_channels=n_input_channels,
            num_classes=num_classes,
            spatial_dims=spatial_dims,
            returned_layers=returned_layers,
            base_anchor_shapes=base_anchor_shapes,
            conv1_t_stride=conv1_t_stride
        )
    else:
        model_path = pretrained_path or "checkpoints/RetinaNet/retinanet_22_epochs.pt"
        net = load_retinanet_from_checkpoint(model_path)

    # Build and configure detector
    detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=debug).to(device)
    return _configure_detector(detector, score_thresh, nms_thresh)
