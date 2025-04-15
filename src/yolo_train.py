from ultralytics import YOLO
from config.config_2d import get_config
import logging
import utils.utils as utils 
import torch
from pathlib import Path
import wandb

if __name__ == "__main__":
    config = get_config()
    logger = logging.getLogger(__name__)
    utils.setup_logging()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = YOLO(model="dev_roi/train4/weights/best.pt", task="detect")
    
    # logger.info(model.model.model)
    
    results = model.train(data="dataset.yaml",
                          epochs=config.epochs,
                          imgsz=512,
                          batch=16,
                          device=device,
                          optimizer="AdamW",
                          seed=config.random_state,
                          single_cls=True,
                          project=config.experiment_name,
                          patience=10
                          )
    
    metrics = model.val(data="dataset.yaml",
                        imgsz=512,
                        batch=16,
                        device=device,
                        project=config.experiment_name,
                        )  # no arguments needed, dataset and settings remembered
    logger.info(f"mAP: {metrics.box.map}") # mAP@50:95
    logger.info(f"mAP50: {metrics.box.map50}")
    logger.info(f"mAP75: {metrics.box.map75}")
    
    # # print all paramaters of metrics.box
    # print(metrics.box)
    
    
    #mAP@5:95 step of 5
    