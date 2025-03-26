import gc
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb



# MONAI imports
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.transforms.box_ops import convert_box_to_mask
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxModed,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    StandardizeEmptyBoxd,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.config import KeysCollection
from monai.data import box_utils
from monai.networks.nets import resnet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    DeleteItemsd,
    apply_transform,
)
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.transforms.utility.dictionary import ApplyTransformToPointsd
from monai.utils.type_conversion import convert_data_type

from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch

# Local imports
from config.config import get_config
from data.dlcs_dataset import DLCSDataset
from data.dlcs_preprocessing import GenerateBoxMask
from models.checkpointed_resnet import CheckpointedResNet
import utils.utils as utils

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")    
    logger.info(f"cwd: {os.getcwd()}")
    data_dir = Path("../DLCS/subset_01_processed")
    annotations_path = Path("../DLCS/DLCSD24_Annotations_voxel.csv")
    
    pretrained = False
    
    base_anchor_shapes = [[6,8,4],[8,6,5],[10,10,6]] # [width, height, depth]
    conv1_t_stride = [2,2,1] # kernel stride for the first conv layer
    returned_layers = [1,2]
    nms_thresh = 0.22
    score_thresh = 0.02
    spatial_dims = 3
    spacing = [0.703125, 0.703125, 1.25] # mm
    n_input_channels = 1
    
    annotations = pd.read_csv(annotations_path)
    logger.info(f"Annotations loaded from {annotations_path}")
        
    image_key = "image"
    box_key = "box"
    label_key = "label"
    point_key = "points"
    label_mask_key = "label_mask"
    box_mask_key = "box_mask"
    
    gt_box_mode = "cccwhd"
    
    affine_lps_to_ras = False
    patch_size = (192, 192, 80)
    batch_size = 4 # more than 4 will cause OOM (on a 16GB GPU)

    train_transform = Compose([
        LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict", reader="NumpyReader"),
        Lambdad(keys=[image_key], func=lambda x: np.transpose(x, (2, 1, 0))),  # D, H, W -> WHD (x, y, z)
        EnsureChannelFirstd(keys=[image_key], channel_dim='no_channel'),
        EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
        EnsureTyped(keys=[label_key], dtype=torch.long),
        StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
        ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
        ConvertBoxToPointsd(keys=[box_key], point_key=point_key),
        GenerateBoxMask(
            keys=box_key,
            image_key=image_key,
            box_key=box_key,
            mask_image_key=label_mask_key,
            spatial_size=patch_size,
        ),
        RandCropByPosNegLabeld(
            keys=[image_key],
            label_key=label_mask_key,
            spatial_size=patch_size,
            num_samples=batch_size,
            pos=1,
            neg=0,
        ),
        ApplyTransformToPointsd(keys=[point_key],
                                refer_keys=image_key,
                                affine_lps_to_ras=affine_lps_to_ras,
                                ),
        ConvertPointsToBoxesd(keys=[point_key], box_key=box_key),
        ClipBoxToImaged(
            box_keys=box_key,
            label_keys=[label_key],
            box_ref_image_keys=image_key,
            remove_empty=True,
        ),
        DeleteItemsd(keys=[label_mask_key, point_key, "image_meta_dict", "image_meta_dict_meta_dict"]),
        EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
    ])
    
    val_transform = Compose([
        LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict", reader="NumpyReader"),
        Lambdad(keys=[image_key], func=lambda x: np.transpose(x, (2, 1, 0))),  # D, H, W -> WHD (x, y, z)
        EnsureChannelFirstd(keys=[image_key], channel_dim='no_channel'),
        EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
        EnsureTyped(keys=[label_key], dtype=torch.long),
        StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
        ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
        EnsureTyped(keys=[image_key, box_key], dtype=torch.float16),
    ])
    
    pids = annotations["patient-id"].unique()
    logger.info(f"Unique patients: {len(pids)}")
    # split into train and val
    train_pids = pids[:int(len(pids)*0.8)]
    val_pids = pids[int(len(pids)*0.8):]
    logger.info(f"Train patients: {len(train_pids)} | Val patients: {len(val_pids)}")
    
    train_annotations = annotations[annotations["patient-id"].isin(train_pids)]
    train_annotations.reset_index(drop=True, inplace=True)
    
    val_annotations = annotations[annotations["patient-id"].isin(val_pids)]
    val_annotations.reset_index(drop=True, inplace=True)
    
    train_ds = DLCSDataset(train_annotations, data_dir, transform=train_transform)
    val_ds = DLCSDataset(val_annotations, data_dir, transform=val_transform)
    
    train_dl = train_ds.get_loader(shuffle=True, num_workers=4)
    val_dl = val_ds.get_loader(shuffle=False, num_workers=4)
    

    # 1) build anchor generator
    # returned_layers: when target boxes are small, set it smaller
    # base_anchor_shapes: anchor shape for the most high-resolution output,
    #   when target boxes are small, set it smaller
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(returned_layers) + 1)], 
        base_anchor_shapes=base_anchor_shapes,
    )

    # 2) build network
    if not pretrained:
        conv1_t_size = [max(7, 2 * s + 1) for s in conv1_t_stride]  # kernel size must be odd, for [2, 2, 1] -> [5, 5, 3]
        backbone = resnet.ResNet(
            block=resnet.ResNetBottleneck,
            layers=[3,4,6,3], # always a list of 4 elements, indicating the number of blocks in each resnet layer
            block_inplanes=resnet.get_inplanes(),
            n_input_channels=n_input_channels,
            conv1_t_stride=conv1_t_stride,
            conv1_t_size=conv1_t_size,
        )
        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=spatial_dims,
            pretrained_backbone=False,
            trainable_backbone_layers=None,
            returned_layers=returned_layers,
        )
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        size_divisible = [s * 2 * 2 ** max(returned_layers) for s in feature_extractor.body.conv1.stride]
        net = torch.jit.script(
            RetinaNet(
                spatial_dims=spatial_dims,
                num_classes=2, # malignant and bening (exclude background)
                num_anchors=num_anchors,
                feature_extractor=feature_extractor,
                size_divisible=size_divisible,
            )
        )
    else:
        logger.info("Loading pretrained model...")
        net = torch.jit.load("checkpoints/RetinaNet/retinanet_test_detector_last.pt")

    # 3) build detector
    detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False).to(device)

    # set training components
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False) # the atss matcher role is to match anchors to gt boxes 
    # detector.set_hard_negative_sampler( 
    #     batch_size_per_image=64,
    #     positive_fraction=balanced_sampler_pos_fraction,
    #     pool_size=20,
    #     min_neg=16,
    # )
    detector.set_target_keys(box_key="box", label_key="label")

    # set validation components
    detector.set_box_selector_parameters(
        score_thresh=score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=nms_thresh,
        detections_per_img=300,
    )
    
    detector.set_sliding_window_inferer(
        roi_size=[512,512,128],
        overlap=0.10,
        sw_batch_size=1,
        mode="constant",
        device="cpu",
    ) 
    
   
    scaler = GradScaler("cuda")
    
    detector.train()
    train_pbar = tqdm(train_dl, total=len(train_dl))
    
    optimizer = torch.optim.SGD(
        detector.network.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True,
    )
    
    # wandb.init(project=self.config.project_name,
    #                    name=self.config.experiment_name,
    #                    notes=self.config.notes,
    #                    tags=[self.model.__class__.__name__,
    #                          f"{self.config.train_split_ratio} train split",
    #                          f"{self.config.val_test_split_ratio} val split",
    #                          self.optimizer.__class__.__name__,
    #                          self.scheduler.__class__.__name__ if self.scheduler is not None else "No scheduler",
    #                          "Pretrained" if self.model.weights is not None else "Not pretrained"])
    #         wandb.watch(self.model)
    
    coco_metric = COCOMetric(classes=["malignant", "benign"], iou_list=[0.1, 0.5, 0.75], iou_range=[0.5, 0.95, 0.05], max_detection=[100])
    optimizer.zero_grad()
    epochs = 10
    # ------------- Training loop -------------
    if not pretrained:
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            for batched_data in train_pbar: # single epoch
                inputs = [
                        batch_data_i["image"].squeeze(0).to(device)
                        for batch_data_i in batched_data
                    ]
                
                logger.debug(f"Inputs type: {type(inputs)} - Single-item type: {type(inputs[0])}")
                logger.debug(f"Inputs shape: {[input.shape for input in inputs]}")
                
                targets = [ # box and labels
                    dict(label=batch_data_i["label"].squeeze(0).to(device),box=batch_data_i["box"].squeeze(0).to(device),)
                    for batch_data_i in batched_data
                ]
                
                logger.debug(f"Targets type: {type(targets)} - Single-item type: {type(targets[0])}")
                logger.debug(f"Targets shape: {[target['box'].shape for target in targets]}")

                for param in detector.network.parameters(): # it's a fancier and faster optimizer.zero_grad
                    param.grad = None 
                    
                if scaler is not None:
                    with autocast("cuda"):
                        outputs = detector(inputs, targets) # image should be HWD, we're using WHD but W=H so we're good
                        loss = outputs[detector.cls_key] + outputs[detector.box_reg_key]
                        
                        logger.debug(f"Box Reg loss: {outputs[detector.box_reg_key].item()} | Cls loss: {outputs[detector.cls_key].item()}")
                        logger.debug(f"Total Loss: {loss.item()}")
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        # optimizer.zero_grad()
                        losses = {"tot": loss.item(), "cls": outputs[detector.cls_key].item(), "reg": outputs[detector.box_reg_key].item()}
                        train_pbar.set_postfix(losses)
                
            # save model
            torch.jit.save(detector.network, "checkpoints/RetinaNet/retinanet_test_detector_last.pt")
    else:
        logger.info("skipping training, using pretrained model")
        # ------------- Validation for model selection -------------
    if True:
        if not pretrained:
            del inputs, batched_data

        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Validating...")
        detector.eval()
        val_outputs_all = []
        val_targets_all = []
        val_pbar = tqdm(val_dl, total=len(val_dl))
        
        # counter = 0
        # val_samples = 3
        
        with torch.no_grad():
            for val_data in val_pbar:
                # if all val_data_i["image"] smaller than args.val_patch_size, no need to use inferer
                # otherwise, need inferer to handle large input images.
                # use_inferer = not all(
                #     [val_data_i["image"][0, ...].numel() < np.prod(args.val_patch_size) for val_data_i in val_data]
                # )
                val_inputs = [val_data.pop("image").squeeze(0).to(device)] # we pop so that val data now contains only boxes and labels
                logger.debug(f"Val inputs type: {type(val_inputs)} - Single-item type: {type(val_inputs[0])}")
                logger.debug(f"Val inputs shape: {[val_input.shape for val_input in val_inputs]}")
                
                if scaler is not None:
                    with torch.autocast("cuda"):
                        val_outputs = detector(val_inputs, use_inferer=True) # only inference

                # save outputs for evaluation
                val_outputs_all += val_outputs 
                val_targets_all += [val_data] # here are the ground truths (without the image)
                
                # counter += 1
                # if counter > val_samples:
                #     break
                
        # TODO: visualize an inference image and boxes
        
        # compute metrics
        del val_inputs
        torch.cuda.empty_cache()
        
        results_metric = matching_batch(
            iou_fn=box_utils.box_iou,
            iou_thresholds=coco_metric.iou_thresholds,
            pred_boxes=[
                val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_outputs_all
            ],
            pred_classes=[
                val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_outputs_all
            ],
            pred_scores=[
                val_data_i[detector.pred_score_key].cpu().detach().numpy() for val_data_i in val_outputs_all
            ],
            gt_boxes=[val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_targets_all],
            gt_classes=[
                val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_targets_all
            ],
        )
        val_epoch_metric_dict = coco_metric(results_metric)[0]
        print(val_epoch_metric_dict)