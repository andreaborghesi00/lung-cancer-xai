from config.config import get_config
from config.config import ModelConfig
from data.preprocessing import ROIPreprocessor
from data.image_dataset import ImageDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import logging
from utils.utils import setup_logging
from models.roi_regressor import RoiRegressor
import torch
from training.trainer import ROITrainer


if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    setup_logging(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # prepare data
    preprocessor = ROIPreprocessor()
    data, labels = preprocessor.load_data_labels()

    # split data
    X_train, X_valtest, y_train, y_valtest = train_test_split(data, labels, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=0)
    logger.info(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
    logger.info(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
    
    # datasets and dataloaders
    logger.info("Creating datasets and dataloaders")
    train_ds = ImageDataset(X_train, y_train)
    val_ds = ImageDataset(X_val, y_val)
    
    logger.info("Creating dataloaders")
    train_dl = train_ds.get_loader(shuffle=True)
    val_dl = val_ds.get_loader()
  
    # model definition
    logger.info("Creating the model")
    model = RoiRegressor().to(device)
      
    # optimizer and scheduler
    logger.info("Initializing optimizer and scheduler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4)
    
    # training
    logger.info("Initializing trainer")
    trainer = ROITrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_dl,
        val_loader=val_dl,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config.checkpoint_dir,
        use_wandb=config.use_wandb
    )
    
    logger.info("Training the model")
    trainer.train(num_epochs=config.epochs)
    
    # testing
    logger.info("Testing the model")
    test_ds = ImageDataset(X_test, y_test)
    test_dl = test_ds.get_loader()
    test_metrics = trainer.validation(test_dl) 
    
    logger.info(f"Test metrics: {test_metrics}")
    
    
    
    
    

