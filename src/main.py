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
    
    # prepare data
    preprocessor = ROIPreprocessor()
    data, labels = preprocessor.load_paths_labels() # no transformation here, i do not have enough ram apparently

    # split data
    X_train, X_valtest, y_train, y_valtest = train_test_split(data, labels, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=0)
    logger.info(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
    logger.info(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
    
    # datasets and dataloaders
    logger.info("Creating datasets and dataloaders")
    train_ds = DynamicRCNNDataset(X_train, y_train, transform=model.get_transform())
    val_ds = DynamicRCNNDataset(X_val, y_val, transform=model.get_transform())
    
    logger.info("Creating dataloaders")
    train_dl = train_ds.get_loader(shuffle=True)
    val_dl = val_ds.get_loader()
  
    # free memory
    del X_train, X_val, y_train, y_val, X_valtest, y_valtest, train_ds, val_ds, data, labels
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
    test_ds = DynamicRCNNDataset(X_test, y_test, transform=model.get_transform())
    test_dl = test_ds.get_loader()
    test_metrics = trainer.validation(test_dl) 
    
    logger.info(f"Test metrics: {test_metrics}")