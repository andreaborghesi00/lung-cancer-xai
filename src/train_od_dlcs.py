import logging
import utils.utils as utils
from config.config_2d import get_config
import torch
from sklearn.model_selection import train_test_split
from data.rcnn_dataset import DynamicResampledDLCS, DynamicResampledDLCSOld
import models.faster_rcnn as frcnn
import models.retinanet as rn
from training.rcnn_trainer import RCNNTrainer
import gc
import pandas as pd
import torch
import numpy as np
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the DLCS dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to the configuration file.")
    parser.add_argument("--model", type=str, default="FasterRCNNEfficientNetv2s", help="Model to use for training.")
    parser.add_argument("--pretrain", type=str, default=None, help="Load pretrained model from checkpoint.")
    
    args = parser.parse_args()
    
    config = get_config(args.config, force_reload=True)
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")    
    
    # model_path = Path(config.checkpoint_dir) / config.model_checkpoint
    # logger.info(f"Loading model from checkpoint from {model_path}")
    # model = utils.load_model(model_path, frcnn.FasterRCNNEfficientNetv2s, device=device)
    
    # model 
    logger.info("Initializing model")
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

    config.experiment_name = f"{model_class.__name__} " + f"{config.experiment_name}"
    if args.pretrain is not None:
        model_path = Path(config.checkpoint_dir) / args.pretrain
        logger.info(f"Loading pretrained model from {model_path}")
        model = utils.load_model(model_path=model_path,
                                 model_class=model_class,
                                 device=device)
        config.experiment_name += f" pretrained"
    else:
        model = model_class(num_classes=2)

    model = model.to(device)
    logger.info(f"Model initialized {model.__class__.__name__} with {utils.count_parameters(model)} trainable parameters")
    
    # prepare data
    logger.info("Loading paths and labels")
    annotations = pd.read_csv(config.annotation_path)
    unique_tomographies = annotations['pid'].unique()
    logger.info(f"Unique tomographies: {len(unique_tomographies)}")
    
    # split unique tomographies into train, validation and test
    logger.info("Preparing data")
    train_ids, valtest_ids = train_test_split(unique_tomographies, test_size=(1-config.train_split_ratio), random_state=config.random_state)
    val_ids, test_ids = train_test_split(valtest_ids, test_size=(1-config.val_test_split_ratio), random_state=config.random_state)

    # def compute_difficulty(df):
    #     df["area"] = (df["bbox_x2"] - df["bbox_x1"]) * (df["bbox_y2"] - df["bbox_y1"])
    #     hu_min  = -1000
    #     hu_max = 500

    #     # Normalize HU and area between 0 and 1
    #     df["hu_norm"] = (df['nodule_mean_intensity'] - hu_min) / (hu_max - hu_min)
    #     df["area_norm"] = (df["area"] - df["area"].min()) / (df["area"].max() - df["area"].min())

    #     # Define difficulty as inverse of ease (higher is harder)
    #     df["difficulty"] = 1 - 0.5 * (df["hu_norm"] + df["area_norm"])
        
    #     # normalize difficulty to [0, 1]
    #     df["difficulty"] = (df["difficulty"] - df["difficulty"].min()) / (df["difficulty"].max() - df["difficulty"].min())
    #     return df
    
    train_annotations = annotations[annotations['pid'].isin(train_ids)].copy()
    # train_annotations = compute_difficulty(train_annotations)    
    
    # # Sampler to handle class imbalance AND curriculum learning, what a move boi
    # sampler = CurriculumSampler(
    #     labels=[0 for _ in range(len(train_annotations))], # dummy labels, we don't need them here
    #     difficulties=train_annotations['difficulty'].tolist(),
    #     total_epochs=6, # reaching full throttle for difficulty at epoch 10
    #     samples_per_epoch=config.batch_size * 600
    # )

    X_train = annotations[annotations['pid'].isin(train_ids)]['path'].values
    X_train = np.array(X_train)
    boxes_train = annotations[annotations['pid'].isin(train_ids)][["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values
    boxes_train = torch.tensor(boxes_train, dtype=torch.float32)
    class_train = annotations[annotations['pid'].isin(train_ids)]['is_benign'].values
    class_train = torch.tensor(class_train, dtype=torch.float32)
    
    X_val = annotations[annotations['pid'].isin(val_ids)]['path'].values
    X_val = np.array(X_val)
    boxes_val = annotations[annotations['pid'].isin(val_ids)][["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values
    boxes_val = torch.tensor(boxes_val, dtype=torch.float32)
    class_val = annotations[annotations['pid'].isin(val_ids)]['is_benign'].values    
    class_val = torch.tensor(class_val, dtype=torch.float32)
    
    logger.info(f"Train data shape: {X_train.shape}, Train labels shape: {boxes_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}, Validation labels shape: {boxes_val.shape}")
    
    
    logger.info("Creating datasets and dataloaders")
    train_ds = DynamicResampledDLCSOld(X_train, boxes_train, class_train, augment=config.augment, annotations=train_annotations)
    val_ds = DynamicResampledDLCSOld(X_val, boxes_val, class_val, augment=False)
    
    # train_dl = train_ds.get_loader(shuffle=False, batch_size=config.batch_size, sampler=sampler)
    train_dl = train_ds.get_loader(shuffle=True, batch_size=config.batch_size)
    val_dl = val_ds.get_loader(batch_size = config.batch_size) 
    
    # free memory
    del X_train, X_val, train_ds, val_ds, unique_tomographies, train_ids, val_ids
    gc.collect() # garbage collection 

    # optimizer and scheduler
    logger.info("Initializing optimizer and scheduler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=2) # max since we are maximizing iou
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0.0)  # to avoid issues with amp when gradients are large
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # simple step scheduler
    
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
    # remove last model and load best model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    best_model_path = Path(config.checkpoint_dir) / f"{model_class.__name__}" / "checkpoint_epoch_best.pt"
    model = utils.load_model(best_model_path, model_class, device=device)
    trainer = RCNNTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config.checkpoint_dir,
        use_wandb=config.use_wandb,
        amp=config.amp,
    )
    
    logger.info("Preparing test data")
    X_test = annotations[annotations['pid'].isin(test_ids)]['path'].values
    X_test = np.array(X_test)
    boxes_test = annotations[annotations['pid'].isin(test_ids)][["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values
    boxes_test = torch.tensor(boxes_test, dtype=torch.float32)
    class_test = annotations[annotations['pid'].isin(test_ids)]['is_benign'].values    
    class_test = torch.tensor(class_test, dtype=torch.float32)
    
    logger.info(f"Test data shape: {X_test.shape}, Test labels shape: {boxes_test.shape}")
    logger.info("Creating test dataset and dataloader")
    test_ds = DynamicResampledDLCSOld(X_test, boxes_test, class_test, augment=False)
    test_dl = test_ds.get_loader(batch_size=config.batch_size)
    logger.info("Testing...")
    metrics, coco_results_dict = trainer.validation(test_dl)
    # upload coco dict to wandb if enabled
    trainer.save_ap_ar_plot(coco_results_dict)
    