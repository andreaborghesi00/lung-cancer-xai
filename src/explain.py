import logging
import random
from config.config import get_config
from data.rcnn_dataset import DynamicRCNNDataset
import utils.utils as utils
from models.faster_rcnn import FasterRCNNMobileNet
from explainers.grad_cam import CAMExplainer
from utils.visualization import Visualizer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from typing import List, Union
from PIL import Image
import numpy as np
from pathlib import Path
from data.preprocessing import ROIPreprocessor
from sklearn.model_selection import train_test_split
import gc

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #load model
    model_path = Path(config.checkpoint_dir) / config.model_checkpoint
    logger.info(f"Loading model from checkpoint from {model_path}")
    model = utils.load_model(model_path, FasterRCNNMobileNet)
    
    # load data
    preprocessor = ROIPreprocessor()
    unique_tomographies = preprocessor.load_tomography_ids()
    _, valtest_ids = train_test_split(unique_tomographies, test_size=(1-config.train_split_ratio), random_state=config.random_state)
    _, test_ids = train_test_split(valtest_ids, test_size=(1-config.val_test_split_ratio), random_state=config.random_state)
    
    X_test, y_test = preprocessor.load_paths_labels(test_ids)
    test_ds = DynamicRCNNDataset(X_test, y_test, transform=model.get_transform())
    test_dl = test_ds.get_loader()
    
    del valtest_ids, test_ids, unique_tomographies
    gc.collect()
    
    visualizer = Visualizer()    

    num_samples = len(test_ds) # all of them
    # sample_indices = random.sample(range(len(test_ds)), num_samples)
    
    logger.info(f"Generating visualizations for {num_samples} random test samples")
    model.eval()
    
    target_layers = [list(model.model.backbone.body.children())[-1]] # get last layer for RCNN model, unfortunately this is model specific
    # cam = GradCAM(model=model, target_layers=target_layers)
    explainer = CAMExplainer(model=model, target_layer=target_layers, target_class=1)

    # apply the explainer to the test samples
    for idx in tqdm(range(num_samples)):
        # Get sample
        image_path = X_test[idx]
        image, target = test_ds[idx]
        image = image.unsqueeze(0).to(device)
        image.requires_grad = True
        
        # Get prediction
        with torch.no_grad():
            prediction = model(image)[0]
        
        # Filter predictions by confidence
        confidence_threshold = 0.5
        mask = prediction['scores'] > confidence_threshold
        pred_boxes = prediction['boxes'][mask]
        scores = prediction['scores'][mask]        
        
        true_boxes = target['boxes'] # ground truth boxes
        
        try:
            grayscale_cam = explainer.explain(image=image, labels=[1], bboxes=[target['boxes'][0].cpu().numpy()])
        except Exception as e:
            logger.error(f"Failed to generate CAM for sample {idx}: {e}")
            continue
            
        image_numpy = image.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0) # convert to numpy and convert from (C, H, W) to (H, W, C)
        visualization = explainer.visualize(image=image_numpy, cam=grayscale_cam[0])
        visualizer.display_bboxes(
            input=visualization,
            pred_boxes=pred_boxes,
            true_boxes=true_boxes,
            scores=scores,
            filename=f"sample_{idx}.png"
        )
    
        # with torch.no_grad():
    #     for idx in tqdm(sample_indices):
    #         # Get sample
    #         image_path = X_test[idx]
    #         image, target = test_ds[idx]
    #         image = image.unsqueeze(0).to(device)  # Add batch dimension
            
    #         # Get prediction
    #         prediction = model(image)[0]
            
    #         # Filter predictions by confidence
    #         confidence_threshold = 0.5
    #         mask = prediction['scores'] > confidence_threshold
    #         pred_boxes = prediction['boxes'][mask]
    #         scores = prediction['scores'][mask]
            
    #         # Get ground truth boxes
    #         true_boxes = target['boxes']
            
    #         # Save visualization
    #         save_path = viz_dir / f"sample_{idx}_comparison.png"
    #         visualize_comparison(
    #             image_path,
    #             pred_boxes,
    #             true_boxes,
    #             scores,
    #             str(save_path)
    #         )
    
    # logger.info(f"Visualizations saved to {viz_dir}")
    