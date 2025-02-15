import logging
import utils.utils as utils
from config.config import get_config
import torch
from sklearn.model_selection import train_test_split
from data.preprocessing import ROIPreprocessor
from data.image_dataset import ImageDataset
from data.rcnn_dataset import StaticRCNNDataset, DynamicRCNNDataset
from models.roi_regressor import RoiRegressor
import models.faster_rcnn as frcnn
from training.trainer import ROITrainer
from training.rcnn_trainer import RCNNTrainer
import gc
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from pathlib import Path

# TODO: move (and refactor) these two functions to a separate package
def _load_model(model_path: str):
    """Load the model from checkpoint"""
    from models.faster_rcnn import FasterRCNNMobileNet  # Import here to avoid circular imports
    
    model = FasterRCNNMobileNet().to("cuda")
    checkpoint = torch.load(model_path, map_location="cuda")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def visualize_comparison(
    image_path: str,
    pred_boxes: torch.Tensor,
    true_boxes: torch.Tensor,
    scores: torch.Tensor,
    save_path: str,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize predicted and ground truth boxes on the same image
    Args:
        image_path: Path to the image
        pred_boxes: Predicted boxes (N, 4) in (x1, y1, x2, y2) format
        true_boxes: Ground truth boxes (M, 4) in (x1, y1, x2, y2) format
        scores: Confidence scores for predicted boxes
        save_path: Path to save the visualization
    """
    # Load and convert image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    plt.figure(figsize=figsize)
    plt.imshow(image_array, cmap='gray')
    
    # Plot predicted boxes in red
    for box, score in zip(pred_boxes, scores):
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = plt.Rectangle(
            (x1, y1), 
            x2 - x1, 
            y2 - y1, 
            fill=False, 
            color='red', 
            linewidth=2,
            label='Predicted' if box is pred_boxes[0] else None  # Label only first box
        )
        plt.gca().add_patch(rect)
        plt.text(
            x1, y1 - 5, 
            f'Pred: {score:.2f}', 
            color='red', 
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    # Plot ground truth boxes in green
    for box in true_boxes:
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = plt.Rectangle(
            (x1, y1), 
            x2 - x1, 
            y2 - y1, 
            fill=False, 
            color='green', 
            linewidth=2,
            label='Ground Truth' if box is true_boxes[0] else None  # Label only first box
        )
        plt.gca().add_patch(rect)
    
    plt.legend()
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    
    # model 
    logger.info("Initializing model")
    model = frcnn.FasterRCNNMobileNet()
    # model = _load_model("checkpoints/checkpoint_epoch_26.pt")
    logger.info(f"Model initialized {model.__class__.__name__} with {utils.count_parameters(model)} trainable parameters")
    model = model.to(device) # maybe this will keep it off the ram
    
    # prepare data
    preprocessor = ROIPreprocessor()
    # data, labels = preprocessor.load_paths_labels() # no transformation here, i do not have enough ram apparently
    unique_tomographies = preprocessor.load_tomography_ids()
    # split unique tomographies into train, validation and test
    logger.info("Splitting data into train, validation and test sets")
    train_ids, valtest_ids = train_test_split(unique_tomographies, test_size=0.2, random_state=0)
    val_ids, test_ids = train_test_split(valtest_ids, test_size=0.5, random_state=0)

    # load paths and labels
    logger.info("Loading paths and labels")
    X_train, y_train = preprocessor.load_paths_labels(train_ids)
    X_val, y_val = preprocessor.load_paths_labels(val_ids)
    logger.info(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
    
    # split data
    # X_train, X_valtest, y_train, y_valtest = train_test_split(data, labels, test_size=0.3, random_state=0)
    # X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=0)
    # logger.info(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
    
    # datasets and dataloaders
    logger.info("Creating datasets and dataloaders")
    train_ds = DynamicRCNNDataset(X_train, y_train, transform=model.get_transform())
    val_ds = DynamicRCNNDataset(X_val, y_val, transform=model.get_transform())
    
    logger.info("Creating dataloaders")
    train_dl = train_ds.get_loader(shuffle=True)
    val_dl = val_ds.get_loader()
  
    # free memory
    del X_train, X_val, y_train, y_val, train_ds, val_ds, unique_tomographies, train_ids, val_ids
    gc.collect() # garbage collection 

    # optimizer and scheduler
    logger.info("Initializing optimizer and scheduler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4) # max since we are maximizing iou
    
    # training
    logger.info("Initializing trainer")
    trainer = RCNNTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_dl,
        val_loader=val_dl,
        device=device,
        checkpoint_dir=config.checkpoint_dir,
        use_wandb=config.use_wandb
    )
    
    logger.info("Training the model")
    trainer.train(num_epochs=config.epochs)
    
    # free memory
    del train_dl, val_dl
    gc.collect() # garbage collection
    
    # testing
    logger.info("Testing the model")
    X_test, y_test = preprocessor.load_paths_labels(test_ids)
    test_ds = DynamicRCNNDataset(X_test, y_test, transform=model.get_transform())
    test_dl = test_ds.get_loader()
    test_metrics = trainer.validation(test_dl) 
    
    logger.info(f"Test metrics: {test_metrics}")
    
    # TODO: refactor this into a separate function in another package
    viz_dir = Path("visualizations/test_samples_2")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get random indices for visualization
    num_samples = len(test_ds)
    sample_indices = random.sample(range(len(test_ds)), num_samples)
    
    logger.info(f"Generating visualizations for {num_samples} random test samples")
    model.eval()
    
    with torch.no_grad():
        for idx in sample_indices:
            # Get sample
            image_path = X_test[idx]
            image, target = test_ds[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction
            prediction = model(image)[0]
            
            # Filter predictions by confidence
            confidence_threshold = 0.5
            mask = prediction['scores'] > confidence_threshold
            pred_boxes = prediction['boxes'][mask]
            scores = prediction['scores'][mask]
            
            # Get ground truth boxes
            true_boxes = target['boxes']
            
            # Save visualization
            save_path = viz_dir / f"sample_{idx}_comparison.png"
            visualize_comparison(
                image_path,
                pred_boxes,
                true_boxes,
                scores,
                str(save_path)
            )
    
    logger.info(f"Visualizations saved to {viz_dir}")
    