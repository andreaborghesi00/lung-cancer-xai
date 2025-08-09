import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import logging
import random
from config.config_2d import get_config
from data.rcnn_dataset import DynamicRCNNDataset, DynamicResampledNLST, DynamicResampledDLCS
from data.tomography_dataset import DynamicTomographyDataset
from training.rcnn_trainer import RCNNTrainer
import utils.utils as utils
from models.faster_rcnn import FasterRCNNMobileNet, FasterRCNNResnet50, FasterRCNNEfficientNetv2s
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
from models.retinanet import RetinaNetResnet50, RetinaNetEfficientNetv2s
import pandas as pd
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

def adaptive_segmenter_score(cam_heatmap, true_boxes):
    """
    like segmenter score but the threshold is adaptive.
    Meaning that the threshold adapts to the area of the bounding box.
    """
    
    pg_cam = np.zeros_like(cam_heatmap)
    
    for bbox in true_boxes:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
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

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    #load model
    model_path = Path(config.checkpoint_dir) / config.model_checkpoint
    logger.info(f"Loading model from checkpoint from {model_path}")

    # model = utils.load_model(model_path, FasterRCNNResnet50, device=device)
    # model = utils.load_model(model_path, FasterRCNNMobileNet, device=device)
    model = utils.load_model(model_path, FasterRCNNEfficientNetv2s, device=device)
    # model = utils.load_model(model_path, RetinaNetResnet50, device=device)
    # model = utils.load_model(model_path, RetinaNetEfficientNetv2s, device=device)
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
    # _, test_ids = train_test_split(valtest_ids, test_size=0.15, random_state=69420)
    
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
    
    # SHARED
    test_dl = test_ds.get_loader(batch_size=1)
    # test_ds_tomo = DynamicTomographyDataset(test_ids, transform=model.get_transform())
    # test_dl_tomo = test_ds_tomo.get_loader(batch_size=1)
    
    del valtest_ids, test_ids, unique_tomographies
    gc.collect()
    
    # test model
    trainer = RCNNTrainer(model=model,
                            optimizer=None,
                            scheduler=None,
                            device=device,
                            checkpoint_dir=config.checkpoint_dir,
                            use_wandb=False # we're not actually training here
                          )
    
    metrics, coco_dict = trainer.validation(test_dl)
    logger.info(f"Validation metrics: {metrics}")
    exit()
    
    visualizer = Visualizer()    


    # logger.info(model.model.backbone)
    # inverse_layer_nums = [1,2,3]
    # target_layers = [[list(model.model.backbone.body.children())[-(i+1)]] for i in inverse_layer_nums] # get last layer for RCNN model, unfortunately this is model specific
    # target_layers = [list(model.model.backbone.fpn.inner_blocks.children())[-1]]
    # target_layers = list(model.model.backbone.fpn.inner_blocks.children()) + list(model.model.backbone.fpn.layer_blocks.children())
    target_layers = [list(model.model.backbone.fpn.layer_blocks.children())[-1]]
    # target_layers = [list(model.model.backbone.body.children())[-1]]
    # target_layers = [model.model.backbone]
    
    logger.info(f"Target layers: {target_layers}")
    # apply the explainer to the test samples
    # explainers = [CAMExplainer(model=model, target_layer=target_layer, cam_class=EigenCAM) for target_layer in target_layers]
    explainer = CAMExplainer(model=model, target_layer=target_layers, cam_class=EigenCAM, reshape_transform=None)
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
            if False:
                try:
                    bboxes = pred_boxes.cpu().numpy()
                    mask = box_to_mask(image_numpy.shape[:2], bboxes)
                    grayscale_cam = explainer.explain(image=image, labels=[1] * len(bboxes), bboxes=bboxes, scores=scores, iou_threshold=iou_threshold)

                    pg_score, top_activations, pg_cam = adaptive_segmenter_score(grayscale_cam[0], true_boxes.cpu().numpy())
                    logger.info(f"Pointing game score: {pg_score}")
                    pg_scores.append(pg_score)
                except Exception as e:
                    logger.error(f"Failed to generate CAM for sample {id}: {str(e)}")
                    continue
                
                visualization = explainer.visualize(image=image_numpy, cam=grayscale_cam[0] ** 2)
                point_vis = explainer.visualize(image=image_numpy, cam=pg_cam)
            
            
            visualizer.display_bboxes(
                input=image_numpy,
                pred_boxes=pred_boxes,
                true_boxes=true_boxes,
                scores=scores,
                filename=f"sample_{id}.png",
                cmap='gray'
            )
            
            
            # visualizer.display_bboxes(
            #     input=visualization,
            #     pred_boxes=pred_boxes,
            #     true_boxes=true_boxes,
            #     scores=scores,
            #     filename=f"sample_{id}.png",
            #     title=f"PG Score: {pg_score:.2f}",
            # )
            
            # visualizer.display_bboxes(
            #     input=point_vis,
            #     pred_boxes=np.array([]),
            #     true_boxes=true_boxes,
            #     scores=np.array([]),
            #     filename=f"sample_{id}_pointgame.png",
            # )
            # pbar.set_postfix({"Avg Segmenter Score": np.array(pg_scores).mean()})
        
        # save pg_scores as numpy
        pg_scores = np.array(pg_scores)
        np.save(f"eigencam_adaptive_segmenter.npy", pg_scores)
        logger.info(f"Average Segmenter Score: {pg_scores / float(len(test_dl))}")