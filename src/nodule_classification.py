import torch
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.nodule_classification_dataset import DLCSNoduleClassificationDataset
from models.nodule_classifiers import Resnet18, MobileNet, EfficientNetv2s
from config.config_2d import get_config
from tqdm import tqdm
import wandb
from torch.optim import Optimizer
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.nn import Module

config = get_config()
DEVICE = torch.device(config.device if torch.cuda.is_available() else "cpu")

def train_epoch(model:Module, optimizer:Optimizer, dl: DataLoader, criterion, scaler:GradScaler):
    global DEVICE, config
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dl, desc="Training Progress", unit="batch")
    
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        with autocast("cuda"):    
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({"loss": loss.item()})
        
        if config.use_wandb:
            wandb.log({"train_loss": loss.item()})
    
    epoch_loss = running_loss / len(dl)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validation_epoch(model, dl, criterion):
    global config, DEVICE
    model.eval()
    
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in tqdm(dl, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_loss /= len(dl)
    val_acc = val_correct / val_total
    
    if config.use_wandb:
        wandb.log({"val_loss": val_loss, "val_accuracy": val_acc})
    
    return val_loss, val_acc
        
def main():
    global config, DEVICE
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load annotations
    annotations_df = pd.read_csv(config.annotation_path)
    
    # split annotations into train and val sets according to the patient id, stratifying over the presence of benign nodules
    pids = annotations_df["pid"].unique().tolist()
    benign_pids = annotations_df[annotations_df['is_benign'] == 1]["pid"].unique().tolist()
    malignant_pids = annotations_df[annotations_df['is_benign'] == 0]["pid"].unique().tolist()
    y = [1 if pid in benign_pids else 0 for pid in pids] # 1 if the patient has at least a benign nodule, 0 otherwise. Used to stratify the split
    print(f"Unique patients: {len(pids)}\nOf which {len(benign_pids)} are benign and {len(malignant_pids)} are malignant")

    # split into train and val
    train_pids, val_pids = train_test_split(pids, test_size=0.2, stratify=y, random_state=config.random_state)

    print(f"Train patients: {len(train_pids)} of which {len([pid for pid in train_pids if pid in benign_pids])} are benign and {len([pid for pid in train_pids if pid in malignant_pids])} are malignant")
    print(f"Val patients: {len(val_pids)} of which {len([pid for pid in val_pids if pid in benign_pids])} are benign and {len([pid for pid in val_pids if pid in malignant_pids])} are malignant")


    train_annotations = annotations_df[annotations_df['pid'].isin(train_pids)].reset_index(drop=True)
    val_annotations = annotations_df[annotations_df['pid'].isin(val_pids)].reset_index(drop=True)

    # Create datasets
    train_dataset = DLCSNoduleClassificationDataset(train_annotations, min_size=64, augment=False, zoom_factor=0.8)
    val_dataset = DLCSNoduleClassificationDataset(val_annotations, min_size=64, augment=False, zoom_factor=0.8)
    
    # Create data loaders
    train_loader = train_dataset.get_loader(batch_size=config.batch_size, shuffle=True, num_workers=config.dl_workers)
    val_loader = val_dataset.get_loader(batch_size=config.batch_size, shuffle=False, num_workers=config.dl_workers)

    # model = NoduleClassifier(num_classes=2)
    # model = NoduleClassifierMobileNet(num_classes=2)
    model = EfficientNetv2s(num_classes=2)
    model.to(DEVICE)
    
    if config.use_wandb:
        wandb.init(project="nodule-classification",
                   config=config,
                   name=f"{model.__class__.__name__} weight decay",
                   note="No augmentation")
        wandb.watch(model, log="all", log_freq=300)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    logger.info(f"Model initialized: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    scaler = GradScaler(enabled=config.amp)
    
    # Training loop
    for epoch in range(config.epochs):
        epoch_loss, epoch_acc = train_epoch(model, optimizer, train_loader, criterion, scaler)
        val_loss, val_acc = validation_epoch(model, val_loader, criterion)        
        scheduler.step()

        if config.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
        
    
    
if __name__ == "__main__":
    main()
