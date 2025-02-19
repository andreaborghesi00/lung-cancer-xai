import logging
import utils.utils as utils
from config.config import get_config
import torch
from sklearn.model_selection import train_test_split
from data.preprocessing import ROIPreprocessor
from data.rcnn_dataset import StaticRCNNDataset, DynamicRCNNDataset
import models.faster_rcnn as frcnn
from training.rcnn_trainer import RCNNTrainer
import gc
from pytorch_grad_cam.utils.image import show_cam_on_image

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    
    # model 
    logger.info("Initializing model")
    model = frcnn.FasterRCNNMobileNet()
    logger.info(f"Model initialized {model.__class__.__name__} with {utils.count_parameters(model)} trainable parameters")
    model = model.to(device) # maybe this will keep it off the ram
    logger.info(model.model)
    
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