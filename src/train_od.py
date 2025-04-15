import logging
import utils.utils as utils
from config.config_2d import get_config
import torch
from sklearn.model_selection import train_test_split
from data.rcnn_preprocessing import ROIPreprocessor
from data.rcnn_dataset import StaticRCNNDataset, DynamicRCNNDataset
from data.tomography_dataset import DynamicTomographyDataset
import models.faster_rcnn as frcnn
from training.rcnn_trainer import RCNNTrainer
import gc
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")    
    
    # model 
    logger.info("Initializing model")
    model = frcnn.FasterRCNNMobileNet()
    logger.info(f"Model initialized {model.__class__.__name__} with {utils.count_parameters(model)} trainable parameters")
    
    
    # if torch.cuda.device_count() > 1:
    #     logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
    #     model = torch.nn.DataParallel(model, [0, 1])  # Multi-GPU support

    model = model.to(device)
    logger.debug(model.module if isinstance(model, torch.nn.DataParallel) else model)
    
    # prepare data
    preprocessor = ROIPreprocessor()
    unique_tomographies = preprocessor.load_tomography_ids()
    
    # split unique tomographies into train, validation and test
    logger.info("Splitting data into train, validation and test sets")
    train_ids, valtest_ids = train_test_split(unique_tomographies, test_size=(1-config.train_split_ratio), random_state=config.random_state)
    val_ids, test_ids = train_test_split(valtest_ids, test_size=(1-config.val_test_split_ratio), random_state=config.random_state)

    # load paths and labels
    logger.info("Loading paths and labels")
    X_train, y_train = preprocessor.load_paths_labels(train_ids)
    X_val, y_val = preprocessor.load_paths_labels(val_ids)
    logger.info(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
    
    # datasets and dataloaders
    logger.info("Creating datasets and dataloaders")
    transform = model.module.get_transform() if isinstance(model, torch.nn.DataParallel) else model.get_transform()
    train_ds = DynamicRCNNDataset(X_train, y_train, transform=transform)
    val_ds = DynamicRCNNDataset(X_val, y_val, transform=transform)
    # val_ds = DynamicTomographyDataset(val_ids, transform=transform)
    
    logger.info("Creating dataloaders")
    train_dl = train_ds.get_loader(shuffle=True, batch_size=config.batch_size)
    val_dl = val_ds.get_loader(batch_size = config.batch_size) # this loader gets whole tomographies, hence the smaller batch size
  
    # free memory
    del X_train, X_val, y_train, y_val, train_ds, val_ds, unique_tomographies, train_ids, val_ids
    gc.collect() # garbage collection 

    # optimizer and scheduler
    logger.info("Initializing optimizer and scheduler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=2) # max since we are maximizing iou
    
    # training
    logger.info("Initializing trainer")
    trainer = RCNNTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,

        device=device,
        checkpoint_dir=config.checkpoint_dir,
        use_wandb=config.use_wandb
    )
    
    logger.info("Training the model")
    trainer.train(num_epochs=config.epochs,
                  patience=config.patience,
                  train_loader=train_dl,
                  val_loader=val_dl,
                  )
    
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
    
    test_ds_tomography = DynamicTomographyDataset(test_ids, transform=model.get_transform())
    test_dl_tomography = test_ds_tomography.get_loader()
    test_metrics_tomography = trainer.tomography_aware_validation(test_dl_tomography)
    logger.info(f"Test metrics tomography: {test_metrics_tomography}")