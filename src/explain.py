import logging
import random
from config.config_2d import get_config
from data.rcnn_dataset import DynamicRCNNDataset, DynamicResampledNLST
from data.tomography_dataset import DynamicTomographyDataset
from training.rcnn_trainer import RCNNTrainer
import utils.utils as utils
from models.faster_rcnn import FasterRCNNMobileNet, FasterRCNNResnet50
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

def pointing_game_score(cam_heatmap, true_boxes):
    
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

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    #load model
    model_path = Path(config.checkpoint_dir) / config.model_checkpoint
    logger.info(f"Loading model from checkpoint from {model_path}")
    model = utils.load_model(model_path, FasterRCNNMobileNet, device=device)
    # model = utils.load_model(model_path, FasterRCNNResnet50, device=device)
    
    logger.info(model.model.backbone)
    # exit()
    # load data
    preprocessor = ROIPreprocessor()
    unique_tomographies = preprocessor.load_tomography_ids()
    _, valtest_ids = train_test_split(unique_tomographies, test_size=(1-config.train_split_ratio), random_state=config.random_state)
    _, test_ids = train_test_split(valtest_ids, test_size=(1-config.val_test_split_ratio), random_state=config.random_state)
    
    X_test, y_test = preprocessor.load_paths_labels(test_ids)
    test_ds = DynamicResampledNLST(X_test, y_test, augment=False)
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
    
    visualizer = Visualizer()    

    # num_samples = len(test_ds) # all of them
    # sample_indices = random.sample(range(len(test_ds)), num_samples)
    
    # logger.info(f"Generating visualizations for {num_samples} random test samples")
    model.model.to(device)
    model.eval()

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
    explainer = CAMExplainer(model=model, target_layer=target_layers, cam_class=SSCAM, reshape_transform=None)
    # explainer = CAMExplainer(model=model, target_layer=target_layers, cam_class=AblationCAM, reshape_transform=fasterrcnn_reshape_transform)
    id = -1
    # cam_scores = []
    # cam_scores_with_misses = []
    # cam_metric = CamMultImageConfidenceChange()
    pbar = tqdm(test_dl, total=len(test_dl))
    pg_scores = 0
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
        try:
            bboxes = pred_boxes.cpu().numpy()
            mask = box_to_mask(image_numpy.shape[:2], bboxes)
            grayscale_cam = explainer.explain(image=image, labels=[1] * len(bboxes), bboxes=bboxes)

            pg_score, top_activations, pg_cam = pointing_game_score(grayscale_cam[0], true_boxes.cpu().numpy())
            logger.info(f"Pointing game score: {pg_score}")
            pg_scores += pg_score
            
            #### INCREASE IN CONFIDENCE #####
            # targets = [FasterRCNNBoxScoreTarget([1] * len(bboxes), bboxes, iou_threshold=0.5)]
            # score_change, vis, perturbed_output = cam_metric(image, 1-grayscale_cam, targets, model, return_visualization=True)
            
            # mask = perturbed_output[0]['scores'] > confidence_threshold
        
            # if np.sum(mask_numpy) == 0:
            #     # take max score
            #     max_idx = np.argmax(perturbed_output[0]['scores'].cpu().numpy())
            #     mask = torch.zeros_like(perturbed_output[0]['scores'], dtype=torch.bool)
            #     mask[max_idx] = True
            
            # perturbed_pred_boxes = perturbed_output[0]['boxes'][mask].cpu().numpy()
            # perturbed_scores = perturbed_output[0]['scores'][mask].cpu().numpy()
            
            # score = score_change[0]
            # logger.info(f"Score change in percent: {score * 100}")
            # vis = vis[0].detach().cpu().numpy().transpose(1, 2, 0)
            # vis = np.repeat(vis, 3, axis=-1) # convert to 3 channels
            # visualizer.display_bboxes(
            #     input=vis,
            #     pred_boxes=perturbed_pred_boxes,
            #     true_boxes=true_boxes,
            #     scores=perturbed_scores,
            #     filename=f"sample_{id}_perturbed.png"
            # )
            ##### END INCREASE IN CONFIDENCE #####
            
            
            
            # score = np.sum((grayscale_cam[0] * mask) > 0) / (np.sum(mask) + 1e-8)
            # cam_scores.append(score)
            # cam_scores_with_misses.append(score)
            # logger.info("Model grad:", next(model.model.parameters()).requires_grad)  # Should be True
        except Exception as e:
            logger.error(f"Failed to generate CAM for sample {id}: {str(e)}")
            # logger.error(f"Image requires_grad: {image.requires_grad}")
            # logger.error("Model grad:", next(model.parameters()).requires_grad)  # Should be True

            # cam_scores_with_misses.append(0)
            continue
            
        visualization = explainer.visualize(image=image_numpy, cam=grayscale_cam[0] ** 2)
        # point_cam = np.zeros_like(grayscale_cam[0])
        # point_cam[tuple(top_activations)] = 1
        point_vis = explainer.visualize(image=image_numpy, cam=pg_cam)
        
        visualizer.display_bboxes(
            input=visualization,
            pred_boxes=pred_boxes,
            true_boxes=true_boxes,
            scores=scores,
            filename=f"sample_{id}.png",
            title=f"PG Score: {pg_score:.2f}",
        )
        
        visualizer.display_bboxes(
            input=point_vis,
            pred_boxes=np.array([]),
            true_boxes=true_boxes,
            scores=np.array([]),
            filename=f"sample_{id}_pointgame.png",
        )
        pbar.set_postfix({"Avg PG Score": pg_scores / (id + 1)})
    
    logger.info(f"Average Pointing Game Score: {pg_scores / float(len(test_dl))}")
    # logger.info(f"CAM Average Scores: {np.mean(cam_scores)}")
    # logger.info(f"CAM Average Scores with misses: {np.mean(cam_scores_with_misses)}")