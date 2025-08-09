import logging
import utils.utils as utils
from config.config_2d import get_config
import torch
from sklearn.model_selection import train_test_split
from data.rcnn_preprocessing import ROIPreprocessor
from data.rcnn_dataset import StaticRCNNDataset, DynamicRCNNDataset, DynamicResampledNLST, DynamicResampledDLCS
from data.tomography_dataset import DynamicTomographyDataset
import models.faster_rcnn as frcnn
import models.retinanet as rn
from training.rcnn_trainer import RCNNTrainer
import gc
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import torch
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the DLCS dataset.")
    parser.add_argument("--config", type=str, default="config/config_2d.yaml", help="Path to the configuration file.")
    parser.add_argument("--model", type=str, default="FasterRCNNEfficientNetv2s", help="Model to use for training.")

    args = parser.parse_args()
    
    config = get_config(args.config, force_reload=True)
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")    
    
    logger.info(f"Configs: {config}")
    logger.info(f"Config path: {args.config}")
    
    # model 
    logger.info("Initializing model")
    if str.lower(args.model) == "fasterrcnnefficientnetv2s":
        config.experiment_name = "FasterRCNN ENv2s " + f" {config.experiment_name}" # append model name to experiment name
        model = frcnn.FasterRCNNEfficientNetv2s(num_classes=2)
    elif str.lower(args.model) == "fasterrcnnmobilenet":
        config.experiment_name = "FasterRCNN MobileNet " + f" {config.experiment_name}"
        model = frcnn.FasterRCNNMobileNet(num_classes=2)
    elif str.lower(args.model) == "fasterrcnnresnet50":
        config.experiment_name = "FasterRCNN ResNet50 " + f" {config.experiment_name}"
        model = frcnn.FasterRCNNResnet50(num_classes=2)
    elif str.lower(args.model) == "retinanetresnet50":
        config.experiment_name = "RetinaNet ResNet50 " + f" {config.experiment_name}"
        model = rn.RetinaNetResnet50(num_classes=2)
    elif str.lower(args.model) == "retinanetefficientnetv2s":
        config.experiment_name = "RetinaNet ENv2s " + f" {config.experiment_name}"
        model = rn.RetinaNetEfficientNetv2s(num_classes=2)
    elif str.lower(args.model) == "retinanetmobilenet":
        config.experiment_name = "RetinaNet MobileNet " + f" {config.experiment_name}"
        model = rn.RetinaNetMobileNet(num_classes=2)
    else:
        raise ValueError(f"Unknown model: {args.model}. Supported models are: FasterRCNN, RetinaNet.")

    logger.info(f"Model initialized {model.__class__.__name__} with {utils.count_parameters(model)} trainable parameters")

    model = model.to(device)
    
    # prepare data
    
    ## NLST
    preprocessor = ROIPreprocessor()
    unique_tomographies = preprocessor.load_tomography_ids()
    
    logger.info(f"Unique tomographies: {len(unique_tomographies)}")
    
    # split unique tomographies into train, validation and test
    logger.info("Splitting data into train, validation and test sets")
    train_ids, valtest_ids = train_test_split(unique_tomographies, test_size=(1-config.train_split_ratio), random_state=config.random_state)
    val_ids, test_ids = train_test_split(valtest_ids, test_size=(1-config.val_test_split_ratio), random_state=config.random_state)

    # load paths and labels
    logger.info("Loading paths and labels")
    
    ## NLST
    X_train, y_train = preprocessor.load_paths_labels(train_ids)
    X_val, y_val = preprocessor.load_paths_labels(val_ids)

    # datasets and dataloaders
    logger.info("Creating datasets and dataloaders")
    # transform = model.get_transform()
     
    ## NLST
    train_ds = DynamicResampledNLST(X_train, y_train, augment=config.augment)
    val_ds = DynamicResampledNLST(X_val, y_val, augment=False)
    # val_ds = DynamicTomographyDataset(val_ids, transform=transform)
    
    logger.info("Creating dataloaders")
    train_dl = train_ds.get_loader(shuffle=True, batch_size=config.batch_size)
    val_dl = val_ds.get_loader(batch_size = config.batch_size) # this loader gets whole tomographies, hence the smaller batch size
  
    # free memory
    del X_train, X_val, train_ds, val_ds, unique_tomographies, train_ids, val_ids
    gc.collect() # garbage collection 

    # optimizer and scheduler
    logger.info("Initializing optimizer and scheduler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=2) # max since we are maximizing iou
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0.0)  # to avoid issues with amp when gradients are large

    # training
    logger.info("Initializing trainer")
    trainer = RCNNTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config.checkpoint_dir,
        use_wandb=config.use_wandb,
        amp=config.amp,
    )
    
    logger.info("Training the model")
    trainer.train(num_epochs=config.epochs,
                  patience=config.patience,
                  train_loader=train_dl,
                  val_loader=val_dl,
                  validate_every=config.validate_every,
                  )
    
    # free memory
    del train_dl, val_dl
    gc.collect() # garbage collection
    
    # testing
    logger.info("Testing the model")
    X_test, y_test = preprocessor.load_paths_labels(test_ids)
    
    test_ds = DynamicResampledNLST(X_test, y_test, augment=False)
    test_dl = test_ds.get_loader(batch_size=config.batch_size)
    
    metrics, coco_results_dict = trainer.validation(test_dl)
    # upload coco dict to wandb if enabled
    trainer.save_ap_ar_plot(coco_results_dict)