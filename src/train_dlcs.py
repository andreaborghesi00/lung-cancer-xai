import gc
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from sklearn.model_selection import train_test_split

# MONAI imports
from monai.data import box_utils
from monai.engines.evaluator import SupervisedEvaluator
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch

# Local imports
from config.config_3d import get_config
from data.dlcs_dataset import DLCSDataset
from data.dlcs_preprocessing import get_train_transforms, get_val_transforms, get_train_transforms_nifti, get_val_transforms_nifti
# from models.checkpointed_resnet import CheckpointedResNet
import utils.utils as utils
import models.monai_retinanet as rn

from pathlib import Path

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    torch.backends.cudnn.benchmark = True 
    torch.set_num_threads(4)

    annotations = pd.read_csv(config.annotations_path)

    affine_lps_to_ras = False

    train_transform = get_train_transforms_nifti(
        augment=config.augment,
        patch_size=config.patch_size,
        batch_size=config.crop_batch_size,
        image_key=config.image_key,
        box_key=config.box_key,
        label_key=config.label_key,
        point_key=config.point_key,
        label_mask_key=config.label_mask_key,
        box_mask_key=config.box_mask_key,
        gt_box_mode=config.gt_box_mode,
    )
    
    val_transform = get_val_transforms_nifti(
        image_key=config.image_key,
        box_key=config.box_key,
        label_key=config.label_key,
        gt_box_mode=config.gt_box_mode,
        )
    
    pids = annotations["patient-id"].unique().tolist()
    benign_pids = annotations[annotations['Malignant_lbl'] == 1]["patient-id"].unique().tolist()
    malignant_pids = annotations[annotations['Malignant_lbl'] == 0]["patient-id"].unique().tolist()
    y = [1 if pid in benign_pids else 0 for pid in pids] # 1 if the patient has at least a benign nodule, 0 otherwise. Used to stratify the split
    logger.info(f"Unique patients: {len(pids)}\nOf which {len(benign_pids)} are benign and {len(malignant_pids)} are malignant")

    # split into train and val
    train_pids, val_pids = train_test_split(pids, test_size=0.2, stratify=y, random_state=config.split_seed)
    
    logger.info(f"Train patients: {len(train_pids)} [{len([i for i in train_pids if i in malignant_pids])} | {len([i for i in train_pids if i in benign_pids])}] | Val patients: {len(val_pids)} [{len([i for i in val_pids if i in malignant_pids])} | {len([i for i in val_pids if i in benign_pids])}]")
    
    train_annotations = annotations[annotations["patient-id"].isin(train_pids)]
    train_annotations.reset_index(drop=True, inplace=True)
    
    val_annotations = annotations[annotations["patient-id"].isin(val_pids)]
    val_annotations.reset_index(drop=True, inplace=True)
    
    train_ds = DLCSDataset(train_annotations, config.data_dir, transform=train_transform)
    val_ds = DLCSDataset(val_annotations, config.data_dir, transform=val_transform)
    
    # train_dl = train_ds.get_loader(shuffle=True, num_workers=config.dl_workers, batch_size=config.dl_batch_size) # careful with batch size
    # val_dl = val_ds.get_loader(shuffle=False, num_workers=config.dl_workers, batch_size=1)

    train_dl = train_ds.get_monai_loader(shuffle=True, num_workers=config.dl_workers, batch_size=config.dl_batch_size) # careful with batch size
    val_dl = val_ds.get_monai_loader(shuffle=False, num_workers=config.dl_workers, batch_size=1)
    
    
    detector = rn.create_retinanet_detector(
        device=device,
        pretrained=config.pretrained,
        pretrained_path=None,
        n_input_channels=config.n_input_channels,
        base_anchor_shapes=config.base_anchor_shapes,
        conv1_t_stride=config.conv1_t_stride,
        num_classes=1
    )
    scaler = GradScaler("cuda", init_scale=config.scaler_init_scale, growth_interval=config.scaler_growth_interval)
    
    detector.train()
    
    optimizer = torch.optim.SGD(
        detector.network.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True,
    )
    
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=50,
    #     gamma=0.1,
    # )
    
    
    # optimizer = torch.optim.NAdam(
    #     params=detector.network.parameters(),
    #     lr=1e-3,
    #     betas=(0.9, 0.999),
    #     weight_decay=3e-5,
    # )
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=1e-5,
    )
        
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs],
    )
    

    if config.use_wandb:
        wandb.init(project=config.project_name,
                        name=config.experiment_name,
                        notes=config.notes,
                        tags=[detector.__class__.__name__,
                                optimizer.__class__.__name__,],
        )
        wandb.watch(detector.network)
        wandb.config.update(config.__dict__)

    # coco_metric = COCOMetric(classes=["malignant", "benign"], iou_list=[0.1, 0.5, 0.75], iou_range=[0.5, 0.95, 0.05], max_detection=[100])
    coco_metric = COCOMetric(classes=["nodule"], iou_list=[0.1, 0.5, 0.75], iou_range=[0.5, 0.95, 0.05], max_detection=[100])
    optimizer.zero_grad()
    
    # ------------- Training loop -------------
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")
        train_pbar = tqdm(train_dl,  total=len(train_dl))
        for batch_data in train_pbar: # single epoch            
            inputs = [
                batch_data_ii["image"].to(device) for batch_data_i in batch_data for batch_data_ii in batch_data_i
            ]
            targets = [
                dict(
                    label=batch_data_ii["label"].to(device),
                    box=batch_data_ii["box"].to(device),
                )
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]
            logger.debug(f"Inputs type: {type(inputs)} - Single-item type: {type(inputs[0])}")
            logger.debug(f"Inputs shape: {[input.shape for input in inputs]}")

            
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
                if not torch.isinf(loss).any() and not torch.isnan(loss).any():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logger.info("Loss is inf or nan, skipping step")
                    continue
                
                # optimizer.zero_grad()
                losses = {"tot": loss.item(), "cls": outputs[detector.cls_key].item(), "reg": outputs[detector.box_reg_key].item()}
                if config.use_wandb:
                    wandb.log(losses)
                train_pbar.set_postfix(losses)
        if config.use_wandb:
            wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step()            
        # save model, optimizer and scheduler
        torch.jit.save(detector.network, os.path.join(config.checkpoint_dir, config.last_model_save_path))
        torch.save(optimizer.state_dict(), os.path.join(config.checkpoint_dir, config.optimizer_save_path))
        torch.save(scheduler.state_dict(), os.path.join(config.checkpoint_dir, config.scheduler_save_path))
        
        # save model on wandb
        if config.use_wandb:
            wandb.save(os.path.join(config.checkpoint_dir, config.last_model_save_path),
                       base_path="checkpoints"
                       )
            # save optimizer and scheduler state
            wandb.save(os.path.join(config.checkpoint_dir, config.optimizer_save_path),
                       base_path="checkpoints"
                       )
            wandb.save(os.path.join(config.checkpoint_dir, config.scheduler_save_path),
                       base_path="checkpoints"
                       )

        # ------------- Validation for model selection -------------
        if (epoch+1) % config.validate_every == 0:
            logger.info("Validating...")
            del inputs, batch_data

            torch.cuda.empty_cache()
            gc.collect()
            detector.eval()
            val_outputs_all = []
            val_targets_all = []
            val_pbar = tqdm(val_dl, total=len(val_dl))
            
            with torch.no_grad():
                for val_data in val_pbar:
                    # val_inputs = [val_data.pop("image").squeeze(0).to(device)] # we pop so that val data now contains only boxes and labels
                    val_inputs = [val_data_i.pop("image").to(device) for val_data_i in val_data]

                    logger.debug(f"Val inputs type: {type(val_inputs)} - Single-item type: {type(val_inputs[0])}")
                    logger.debug(f"Val inputs shape: {[val_input.shape for val_input in val_inputs]}")
                    
                    if scaler is not None:
                        with torch.autocast("cuda"):
                            val_outputs = detector(val_inputs, use_inferer=True) # only inference

                    # save outputs for evaluation
                    val_outputs_all += val_outputs 
                    val_targets_all += val_data # here are the ground truths (without the image)
                    
                    
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
            if config.use_wandb:
                wandb.log(val_epoch_metric_dict)
            logger.info(f"Validation metrics: {val_epoch_metric_dict}")
            
            
            
            gc.collect()
            detector.train()
