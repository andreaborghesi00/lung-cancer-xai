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

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")    
    
    # model 
    logger.info("Initializing model")
    # model = frcnn.FasterRCNNMobileNet()
    # model = frcnn.FasterRCNNEfficientNetv2s()
    # model = frcnn.FasterRCNNEfficientNetB2()
    # model = frcnn.FasterRCNNResnet50()
    # model = rn.RetinaNetResnet50(num_classes=2)
    model = rn.RetinaNetEfficientNetv2s(num_classes=2)
    logger.info(f"Model initialized {model.__class__.__name__} with {utils.count_parameters(model)} trainable parameters")

    model = model.to(device)
    
    # prepare data
    

    ## DLCS
    annotations = pd.read_csv(config.annotation_path)
    unique_tomographies = annotations['pid'].unique()
    
    logger.info(f"Unique tomographies: {len(unique_tomographies)}")
    
    # split unique tomographies into train, validation and test
    logger.info("Splitting data into train, validation and test sets")
    train_ids, valtest_ids = train_test_split(unique_tomographies, test_size=(1-config.train_split_ratio), random_state=config.random_state)
    val_ids, test_ids = train_test_split(valtest_ids, test_size=(1-config.val_test_split_ratio), random_state=config.random_state)

    # load paths and labels
    logger.info("Loading paths and labels")
    

    ## DLCS
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
    
    # datasets and dataloaders
    logger.info("Creating datasets and dataloaders")
    # transform = model.get_transform()
     
  
    ## DLCS
    # train_ds = DynamicResampledDLCS(X_train, boxes_train, class_train, augment=config.augment, transform=model.get_transform())
    # val_ds = DynamicResampledDLCS(X_val, boxes_val, class_val, augment=False, transform=model.get_transform())
    
    train_ds = DynamicResampledDLCS(X_train, boxes_train, class_train, augment=config.augment)
    val_ds = DynamicResampledDLCS(X_val, boxes_val, class_val, augment=False)
    
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
    # logger.info("Testing the model")
    # X_test, y_test = preprocessor.load_paths_labels(test_ids)
    # test_ds = DynamicRCNNDataset(X_test, y_test, transform=model.get_transform())
    # test_dl = test_ds.get_loader()
    # test_metrics = trainer.validation(test_dl) 
    # logger.info(f"Test metrics: {test_metrics}")
    
    # test_ds_tomography = DynamicTomographyDataset(test_ids, transform=model.get_transform(), augment=False)
    # test_dl_tomography = test_ds_tomography.get_loader()
    # test_metrics_tomography = trainer.validation(test_dl_tomography)
    # logger.info(f"Test metrics tomography: {test_metrics_tomography}")