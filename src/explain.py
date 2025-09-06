import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import logging
import random
from config.config_2d import get_config
from data.rcnn_dataset import DynamicRCNNDataset, DynamicResampledNLST, DynamicResampledDLCS, DynamicResampledDLCSOld
from training.rcnn_trainer import RCNNTrainer
import utils.utils as utils
from explainers.cam_explainer import CAMExplainer, CustomGradCAM, fasterrcnn_reshape_transform, SSCAM, FasterRCNNBoxScoreTarget
from utils.visualization import Visualizer
from tqdm import tqdm
import torch
from pathlib import Path
from data.rcnn_preprocessing import ROIPreprocessor
from sklearn.model_selection import train_test_split
import gc
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence, CamMultImageConfidenceChange
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd
import argparse
import models.faster_rcnn as frcnn
import models.retinanet as rn
from torch.utils.data import DataLoader
def _get_image_from_tomo(tomo, idx):
    image = tomo[idx].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return image

def box_to_mask(image_shape, bboxes):
    mask = np.zeros(image_shape, dtype=np.uint8)
    if len(bboxes.shape) > 1:
        for bbox in bboxes:
            x,y,w,h = bbox.astype(int)
            mask[y:y+h, x:x+w] = 1
        return mask
    x,y,w,h = bboxes.astype(int)
    mask[y:y+h, x:x+w] = 1
    return mask

def segmenter_score(cam_heatmap, true_boxes):
    
    threshold = np.percentile(cam_heatmap, 99.5) # top 0.5% surviving
    top_activations = np.where(cam_heatmap >= threshold)
    
    points_inside = 0
    tot_points = len(top_activations[0])
    pg_cam = np.zeros_like(cam_heatmap)
    
    for i in range(tot_points):
        y, x = top_activations[0][i], top_activations[1][i]
        for bbox in true_boxes:
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                points_inside += 1
                pg_cam[y, x] = 90
                break
        if pg_cam[y, x] == 0:
            pg_cam[y, x] = 1
    return points_inside / tot_points if tot_points > 0 else 0, top_activations, pg_cam

def adaptive_segmenter_score(cam_heatmap, true_boxes, adaptive_multiplier=1.):
    """
    like segmenter score but the threshold is adaptive.
    Meaning that the threshold adapts to the area of the bounding box.
    """
    
    pg_cam = np.zeros_like(cam_heatmap)
    
    for bbox in true_boxes:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        area += adaptive_multiplier * area # increase area by a multiplier to account for small boxes and noise
        # find the percentile of the cam heatmap that corresponds to the area of the bounding box
        cam_area = cam_heatmap.shape[0] * cam_heatmap.shape[1]
        percentile = (area / float(cam_area)) * 100
        threshold = np.percentile(cam_heatmap, 100 - percentile) 
        top_activations = np.where(cam_heatmap >= threshold)
        points_inside = 0
        tot_points = len(top_activations[0])
        
        for i in range(tot_points):
            y, x = top_activations[0][i], top_activations[1][i]
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                points_inside += 1
                pg_cam[y, x] = 90
                continue
            if pg_cam[y, x] == 0:
                pg_cam[y, x] = 1
    return points_inside / tot_points if tot_points > 0 else 0, top_activations, pg_cam

import numpy as np
import math

def calculate_normalized_inverse_distance_score(cam_heatmap,
                                                true_boxes,
                                                adaptive_multiplier=1.,
                                                norm_order=2):
    """
    Calculates a score based on the normalized distance of the most activated pixels to the ground-truth box.

    This metric is more robust to noise than a simple in/out check. It uses an adaptive
    threshold to select a number of top pixels proportional to the ground-truth box area.

    Args:
        cam_heatmap (np.ndarray): The 2D CAM activation map.
        true_boxes (list): A list of bounding boxes, e.g., [[x1, y1, x2, y2]]. 
                           This implementation considers only the first box.
        adaptive_multiplier (float): A factor to increase the considered area, accounting
                                     for small localization errors. A value of 1.0 doubles it.

    Returns:
        float: A score between 0 and 1. Higher is better, with 1 meaning all considered
               pixels are inside the ground-truth box.
        tuple: The coordinates of the top activated pixels considered for the score.
    """
    # if not true_boxes:
    #     return 0.0, ([], [])

    pg_cam = np.zeros_like(cam_heatmap)
    # We assume one box per image as specified
    bbox = true_boxes[0]
    x1, y1, x2, y2 = bbox
    
    # --- Adaptive Selection of Top Pixels (from your original code) ---
    box_area = (x2 - x1) * (y2 - y1)
    # The multiplier helps select a slightly larger area of pixels to be more robust
    target_area = box_area + (adaptive_multiplier * box_area)
    
    cam_area = cam_heatmap.shape[0] * cam_heatmap.shape[1]
    # Ensure percentile is within a valid range [0, 100]
    percentile = min(100, max(0, (target_area / float(cam_area)) * 100))
    
    if percentile == 0:
        return 0.0, ([], [])

    threshold = np.percentile(cam_heatmap, 100 - percentile)
    top_activations = np.where(cam_heatmap >= threshold)

    if len(top_activations[0]) == 0:
        return 0.0, ([], [])
        
    # --- New Distance-Based Scoring Logic ---
    h, w = cam_heatmap.shape
    image_diagonal = math.sqrt(h**2 + w**2)
    
    normalized_distances = []
    
    # Iterate over each of the top activated pixels
    for y, x in zip(*top_activations):
        # Find the closest point on the bounding box to the current pixel (x, y)
        closest_x = np.clip(x, x1, x2)
        closest_y = np.clip(y, y1, y2)
        
        # The distance is 0 if the point is inside the box
        if norm_order == 2:
            distance = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
        else:
            dx = np.abs(x - closest_x)
            dy = np.abs(y - closest_y)        
            distance = np.power(np.power(dx, norm_order) + np.power(dy, norm_order), 1. / norm_order)
        
        # Normalize the distance by the image diagonal
        normalized_distance = distance / image_diagonal
        normalized_distances.append(normalized_distance)
        
        # For visualization purposes
        if distance == 0:
            pg_cam[y, x] = 90
        else:
            pg_cam[y, x] = 1
        
    # The score is 1 minus the average normalized distance
    # This rewards pixels for being close to or inside the box
    score = 1.0 - np.mean(normalized_distances)
    
    return score, top_activations, pg_cam

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the DLCS dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to the configuration file.")
    parser.add_argument("--model", type=str, default="FasterRCNNEfficientNetv2s", help="Model to use for training.")
    parser.add_argument("--pretrain", type=str, default="FasterRCNNEfficientNetv2s/checkpoint_epoch_best.pt", help="Load pretrained model from checkpoint.")
    parser.add_argument("--cam", type=str, default="eigencam", help="Type of CAM to use for explanation")
    
    args = parser.parse_args()
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    #nohup parallel --colsep '\t' --joblog joblog_dlcs_pretrained.log -j1 python src/train_od_dlcs.py --config=src/config/config_2d.yaml --model {1} --pretrain {2} :::: pairs_models_pretraining_remainder.txt > parallel_dlcs_pretrain_remainder.log 2>&1 &
    #load model
    model_class = None
    if str.lower(args.model) == "fasterrcnnefficientnetv2s":
        model_class = frcnn.FasterRCNNEfficientNetv2s
    elif str.lower(args.model) == "fasterrcnnmobilenet":
        model_class = frcnn.FasterRCNNMobileNet
    elif str.lower(args.model) == "fasterrcnnresnet50":
        model_class = frcnn.FasterRCNNResnet50
    elif str.lower(args.model) == "retinanetresnet50":
        model_class = rn.RetinaNetResnet50
    elif str.lower(args.model) == "retinanetefficientnetv2s":
        model_class = rn.RetinaNetEfficientNetv2s
    elif str.lower(args.model) == "retinanetmobilenet":
        model_class = rn.RetinaNetMobileNet
    else:
        raise ValueError(f"Unknown model: {args.model}. Supported models are: FasterRCNN, RetinaNet.")

    cam_class = None
    if str.lower(args.cam) == "eigencam":
        cam_class = EigenCAM
    elif str.lower(args.cam) == "scorecam":
        cam_class = ScoreCAM
    elif str.lower(args.cam) == "sscam":
        cam_class = SSCAM

    config.visualization_experiment_name += str.lower(args.cam) + "/"
    config.visualization_experiment_name += str.lower(model_class.__name__) + "/"
    
    try:
        model_path = Path(config.checkpoint_dir) / args.pretrain
        logger.info(f"Loading model from checkpoint from {model_path}")

        model = utils.load_model(model_path, model_class, device=device)
    except FileNotFoundError:
        logger.info(f"No pretrained model found in {config.checkpoint_dir} at {args.pretrain}. \nStopping execution")
        exit(1)
            
    model.eval()
    logger.info(model.model.backbone)

    # load data
    # DLCS
    annotations = pd.read_csv(config.annotation_path)
    unique_tomographies = annotations['pid'].unique()
    
    # NLST
    # preprocessor = ROIPreprocessor()
    # unique_tomographies = preprocessor.load_tomography_ids()
    
    # SHARED
    _, valtest_ids = train_test_split(unique_tomographies, test_size=(1-config.train_split_ratio), random_state=config.random_state)
    _, test_ids = train_test_split(valtest_ids, test_size=(1-config.val_test_split_ratio), random_state=config.random_state)
    
    # NLST
    # X_test, y_test = preprocessor.load_paths_labels(test_ids)
    # test_ds = DynamicResampledNLST(X_test, y_test, augment=False)
    
    # DLCS
    X_test = annotations[annotations['pid'].isin(test_ids)]['path'].values
    X_test = np.array(X_test)
    boxes_test = annotations[annotations['pid'].isin(test_ids)][["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values
    boxes_test = torch.tensor(boxes_test, dtype=torch.float32)
    class_test = annotations[annotations['pid'].isin(test_ids)]['is_benign'].values    
    class_test = torch.tensor(class_test, dtype=torch.float32)
    test_ds = DynamicResampledDLCS(X_test, boxes_test, class_test, augment=False)
    # subset_test_ds = torch.utils.data.Subset(test_ds, list(range(0, len(test_ds), 10))) # subsample for faster testing
    # SHARED
    test_dl = test_ds.get_loader(batch_size=1)
    # test_dl = DataLoader(subset_test_ds,
                        #  batch_size=1,
                        #  shuffle=False,
                        #  num_workers=4,
                        #  collate_fn=lambda x: tuple(zip(*x))
                        #  )
    # test_ds_tomo = DynamicTomographyDataset(test_ids, transform=model.get_transform())
    # test_dl_tomo = test_ds_tomo.get_loader(batch_size=1)
    
    del valtest_ids, test_ids, unique_tomographies
    gc.collect()
    
    # test model
    # trainer = RCNNTrainer(model=model,
    #                         optimizer=None,
    #                         scheduler=None,
    #                         device=device,
    #                         checkpoint_dir=config.checkpoint_dir,
    #                         use_wandb=False # we're not actually training here
    #                       )
    
    # metrics, coco_dict = trainer.validation(test_dl)
    # logger.info(f"Validation metrics: {metrics}")
    # exit()
    
    visualizer = Visualizer()    


    # logger.info(model.model.backbone)
    # inverse_layer_nums = [1,2,3]
    # target_layers = [[list(model.model.backbone.body.children())[-(i+1)]] for i in inverse_layer_nums] # get last layer for RCNN model, unfortunately this is model specific
    target_layers = [list(model.model.backbone.fpn.inner_blocks.children())[-1], list(model.model.backbone.fpn.inner_blocks.children())[-2]]
    # target_layers = list(model.model.backbone.fpn.inner_blocks.children()) + list(model.model.backbone.fpn.layer_blocks.children())
    # target_layers = [list(model.model.backbone.fpn.layer_blocks.children())[-1]]
    # target_layers = [list(model.model.backbone.body.children())[-1]]
    # target_layers = [model.model.backbone]
    
    logger.info(f"Target layers: {target_layers}")
    # apply the explainer to the test samples
    # explainers = [CAMExplainer(model=model, target_layer=target_layer, cam_class=EigenCAM) for target_layer in target_layers]
    explainer = CAMExplainer(model=model, target_layer=target_layers, cam_class=cam_class, reshape_transform=None)
    # explainer = CAMExplainer(model=model, target_layer=target_layers, cam_class=AblationCAM, reshape_transform=fasterrcnn_reshape_transform)

    # iou_thresholds = np.arange(0.0, 1.0, 0.1)
    iou_thresholds = [.5]

    for iou_num, iou_threshold in enumerate(iou_thresholds):
        id = -1

        pbar = tqdm(test_dl, total=len(test_dl))
        pg_scores = []
        for image, target in pbar:
            id +=1
            
            image = torch.stack(image).to(device) # add batch dimension
            image.requires_grad = True # this is needed for the CAM explainer
            
            # get prediction to display predicted boxes
            with torch.no_grad():
                prediction = model(image)[0]
            
            # filter predictions by confidence
            confidence_threshold = 0.5
            mask = prediction['scores'] > confidence_threshold 
            
            mask_numpy = mask.cpu().numpy()
            
            if len(mask) == 0:
                logger.error(f"No predictions found for sample {id}")
                continue
            
            if np.sum(mask_numpy) == 0:
                # take max score
                max_idx = np.argmax(prediction['scores'].cpu().numpy())
                mask = torch.zeros_like(prediction['scores'], dtype=torch.bool)
                mask[max_idx] = True
            
            pred_boxes = prediction['boxes'][mask]
            scores = prediction['scores'][mask]        
            
            true_boxes = target[0]['boxes'] # ground truth boxes
            image_numpy = image.detach().squeeze(0).cpu().numpy()
            image_numpy = image_numpy.transpose(1,2,0) # convert to numpy and convert from (C, H, W) to (H, W, C)
            # if False:
            try:
                bboxes = pred_boxes.cpu().numpy()
                mask = box_to_mask(image_numpy.shape[:2], bboxes)
                grayscale_cam = explainer.explain(image=image, labels=[1] * len(bboxes), bboxes=bboxes, scores=scores, iou_threshold=iou_threshold)

                # pg_score, top_activations, pg_cam = adaptive_segmenter_score(grayscale_cam[0], true_boxes.cpu().numpy())
                pg_score, top_activations, pg_cam = calculate_normalized_inverse_distance_score(grayscale_cam[0],
                                                                                                true_boxes.cpu().numpy(),
                                                                                                adaptive_multiplier=1.8,
                                                                                                norm_order=6)
                logger.info(f"Inverse distance score: {pg_score}")
                pg_scores.append(pg_score)
            except Exception as e:
                logger.error(f"Failed to generate CAM for sample {id}: {str(e)}")
                continue
            
            # image_numpy = image_numpy[:,:,1] # take only the center slice (current)
            # image_numpy = image_numpy[:, :, np.newaxis] # add channel dimension
            visualization = explainer.visualize(image=image_numpy, cam=grayscale_cam[0] ** 2)
            point_vis = explainer.visualize(image=image_numpy, cam=pg_cam)
            
            
            # visualizer.display_bboxes(
            #     input=visualization, # display only the center slice (current)
            #     pred_boxes=pred_boxes,
            #     true_boxes=true_boxes,
            #     scores=scores,
            #     filename=f"sample_{id}.png",
            #     cmap='gray'
            # )
        # exit()
            
            visualizer.display_bboxes(
                input=visualization,
                pred_boxes=pred_boxes,
                true_boxes=true_boxes,
                scores=scores,
                filename=f"sample_{id}.png",
                title=f"Score: {pg_score:.2f}",
            )
            
            visualizer.display_bboxes(
                input=point_vis,
                pred_boxes=np.array([]),
                true_boxes=true_boxes,
                scores=np.array([]),
                filename=f"sample_{id}_segmenter.png",
            )
            pbar.set_postfix({"Avg Score": np.array(pg_scores).mean()})
        
        # save pg_scores as numpy
        pg_scores = np.array(pg_scores)
        np.save(Path(config.visualization_dir) / f"{cam_class.__name__}_{model_class.__name__}_distance_scores_2x.npy", pg_scores)
        logger.info(f"Average Score: {np.sum(pg_scores) / float(len(test_dl))}")