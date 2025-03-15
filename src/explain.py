import logging
import random
from config.config import get_config
from data.rcnn_dataset import DynamicRCNNDataset
from data.tomography_dataset import DynamicTomographyDataset
from training.rcnn_trainer import RCNNTrainer
import utils.utils as utils
from models.faster_rcnn import FasterRCNNMobileNet, FasterRCNNResnet50
from explainers.grad_cam import CAMExplainer
from utils.visualization import Visualizer
from tqdm import tqdm
import torch
from pathlib import Path
from data.rcnn_preprocessing import ROIPreprocessor
from sklearn.model_selection import train_test_split
import gc
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
import numpy as np

def _get_image_from_tomo(tomo, idx):
    image = tomo[idx].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return image

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #load model
    model_path = Path(config.checkpoint_dir) / config.model_checkpoint
    logger.info(f"Loading model from checkpoint from {model_path}")
    model = utils.load_model(model_path, FasterRCNNMobileNet, device=device)
    
    # load data
    preprocessor = ROIPreprocessor()
    unique_tomographies = preprocessor.load_tomography_ids()
    _, valtest_ids = train_test_split(unique_tomographies, test_size=(1-config.train_split_ratio), random_state=config.random_state)
    _, test_ids = train_test_split(valtest_ids, test_size=(1-config.val_test_split_ratio), random_state=config.random_state)
    
    X_test, y_test = preprocessor.load_paths_labels(test_ids)
    test_ds = DynamicRCNNDataset(X_test, y_test, transform=model.get_transform(), augment=False)
    test_dl = test_ds.get_loader()
    test_ds_tomo = DynamicTomographyDataset(test_ids, transform=model.get_transform())
    test_dl_tomo = test_ds_tomo.get_loader(batch_size=1)
    
    del valtest_ids, test_ids, unique_tomographies
    gc.collect()
    
    # test model
    trainer = RCNNTrainer(model=model,
                            optimizer=None,
                            scheduler=None,
                            device=device,
                            checkpoint_dir=config.checkpoint_dir,
                            use_wandb=config.use_wandb
                          )
    
    logger.info("Testing model - Tomography aware")
    metrics = trainer.validation_outlier_detection(test_dl_tomo)
    logger.info(f"Test metrics: {metrics}")
    
    logger.info("Testing model - Non tomography aware")
    metrics = trainer.validation(test_dl)
    logger.info(f"Test metrics: {metrics}")
    
    exit()
    
    visualizer = Visualizer()    

    num_samples = len(test_ds) # all of them
    # sample_indices = random.sample(range(len(test_ds)), num_samples)
    
    logger.info(f"Generating visualizations for {num_samples} random test samples")
    model.eval()
    
    # logger.info(model.model.backbone)
    # inverse_layer_nums = [1]
    # target_layers = [[list(model.model.backbone.body.children())[-(i+1)]] for i in inverse_layer_nums] # get last layer for RCNN model, unfortunately this is model specific
    # # apply the explainer to the test samples
    # explainers = [CAMExplainer(model=model, target_layer=target_layer, target_class=1, cam_class=GradCAM) for target_layer in target_layers]
    # logger.info(f"Explainers initialized, length: {len(explainers)}")
    id = 0
    for tomos, targets in tqdm(test_dl_tomo):
        tomos = torch.stack(tomos).to(device)

        results, original, formatted_targets = trainer._process_tomography_with_outlier_detection(tomos, targets)
        confidence_threshold = 0.5
        skips = [skip for _, _, skip in results]
        pred_boxes = [pred['boxes'][pred['scores'] > confidence_threshold] for pred, _, _ in results]
        scores = [pred['scores'][pred['scores'] > confidence_threshold] for pred, _, _ in results]
        true_boxes = [target[0]['boxes'] for target in formatted_targets]
        
        visualizer.display_tomo_bboxes(
            tomo = tomos[0], # i don't like this
            boxes = pred_boxes,
            true_boxes = true_boxes,
            scores = scores,
            save_dir = Path(str(id)),
            skips = skips
        )
        id +=1
        
        
    #     # get prediction to display predicted boxes
    #     with torch.no_grad():
    #         prediction = model(image)[0]
        
    #     # filter predictions by confidence
    #     confidence_threshold = 0.5
    #     mask = prediction['scores'] > confidence_threshold
    #     pred_boxes = prediction['boxes'][mask]
    #     scores = prediction['scores'][mask]        
        
    #     true_boxes = target['boxes'] # ground truth boxes
        
    #     image_numpy = image.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0) # convert to numpy and convert from (C, H, W) to (H, W, C)
    #     for layer_idx, target_layer in enumerate(target_layers):
    #         explainer = explainers[layer_idx]
    #         try:
    #             grayscale_cam = explainer.explain(image=image, labels=[1], bboxes=[target['boxes'][0].cpu().numpy()])
    #         except Exception as e:
    #             logger.error(f"Failed to generate CAM for sample {idx} at layer {layer_idx}: {str(e)}")
    #             continue
                
    #         visualization = explainer.visualize(image=image_numpy, cam=grayscale_cam[0])
    #         visualizer.display_bboxes(
    #             input=visualization,
    #             pred_boxes=pred_boxes,
    #             true_boxes=true_boxes,
    #             scores=scores,
    #             filename=f"sample_{idx}_L{layer_idx}.png"
    #         )
    
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
    