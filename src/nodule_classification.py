import torch
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.nodule_classification_dataset import DLCSNoduleClassificationDataset
from models.nodule_classifiers import Resnet18, MobileNet, EfficientNetv2s, DenseNet121, ConvNeXtTiny
from config.config_2d import get_config
from tqdm import tqdm
import wandb
from torch.optim import Optimizer
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.nn import Module
from typing import Union, Dict
from pathlib import Path
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils.focal_loss import FocalLoss
from data.samplers import CurriculumBalancedSampler
import argparse
from sklearn.metrics import f1_score
import utils.utils as utils

config = get_config()
DEVICE = torch.device(config.device if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_checkpoint(model:Module, optimizer:Optimizer, scheduler:LRScheduler, name: Union[int, str], metrics: Dict[str, float]):
        """
        Save the model checkpoint
        """
        global config, logger
        checkpoint_dir = Path(config.checkpoint_dir) / "nodule_classification" / model.__class__.__name__
        if checkpoint_dir is None:
            return 
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "timestamp": name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        checkpoint_path = checkpoint_dir / model.__class__.__name__ / f"checkpoint_{name}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        if config.use_wandb:
            wandb.save(checkpoint_path, base_path=checkpoint_dir / model.__class__.__name__ )
        
        logger.info(f"Checkpoint saved at {checkpoint_path}")

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
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels.view(-1).float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) >= 0.5).long()  # Convert logits to binary predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({"loss": loss.item()})
        
        if config.use_wandb:
            wandb.log({"train_loss": loss.item()})
    
    epoch_loss = running_loss / len(dl)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validation_epoch(model:Module, dl:DataLoader, criterion, upload_cm:bool=False):
    global config, DEVICE
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Store all labels and predictions for confusion matrix
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dl, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with autocast("cuda"):
                outputs = model(images).view(-1)
                loss = criterion(outputs, labels.view(-1).float())
            
            val_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) >= 0.5).long()  # Convert logits to binary predictions
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(dl)
    val_acc = val_correct / val_total

    # Compute confusion matrix
    all_labels = ["benign" if label == 1 else "malignant" for label in all_labels]
    all_preds = ["benign" if pred == 1 else "malignant" for pred in all_preds]
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    # set the labels to be "malignant" and "benign"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    cm_norm_path = Path(f"confusion_matrix_{model.__class__.__name__}_val_norm.png")
    plt.savefig(cm_norm_path)  # save to file
    plt.close()
    if upload_cm and config.use_wandb:
        wandb.log({"cm_norm": wandb.Image(str(cm_norm_path))})
    
    
    cm = confusion_matrix(all_labels, all_preds, labels=["malignant", "benign"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    cm_path = Path(f"confusion_matrix_{model.__class__.__name__}_val.png")
    plt.savefig(cm_path)  # save to file
    plt.close()
    
    if upload_cm and config.use_wandb:
        wandb.log({"cm": wandb.Image(str(cm_path))})
    
    # Compute F1 scores for each class
    f1_malignant = f1_score(all_labels, all_preds, pos_label="malignant")
    f1_benign = f1_score(all_labels, all_preds, pos_label="benign")
    f1_avg = (f1_malignant + f1_benign) / 2.0
    if config.use_wandb:
        wandb.log({"f1_malignant": f1_malignant, "f1_benign": f1_benign, "f1_avg": f1_avg})
        
    return val_loss, val_acc, f1_avg
        
def main():
    global config, DEVICE, logger
    
    parser = argparse.ArgumentParser(description="Nodule Classification Training")
    parser.add_argument("--model", type=str, default="ConvNeXtTiny", help="Model architecture to use.")
    parser.add_argument("--square_patches", type=int, default=1, help="Whether to use square patches (1) or not (0).")
    
    args = parser.parse_args()
    
    model_class = None
    if str.lower(args.model) == "resnet18":
        model_class = Resnet18
    elif str.lower(args.model) == "mobilenet":
        model_class = MobileNet
    elif str.lower(args.model) == "efficientnetv2s":
        model_class = EfficientNetv2s
    elif str.lower(args.model) == "densenet121":
        model_class = DenseNet121
    elif str.lower(args.model) == "convnexttiny":
        model_class = ConvNeXtTiny
    else:
        raise ValueError(f"Unknown model: {args.model}. Supported models are: Resnet18, MobileNet, EfficientNetv2s, DenseNet121, ConvNeXtTiny.")
    
    square_patches = bool(args.square_patches)
    
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

    # Weights to balance the classes
    benign_count = len(train_annotations[train_annotations['is_benign'] == 1])
    malignant_count = len(train_annotations[train_annotations['is_benign'] == 0])
    total_count = benign_count + malignant_count
    benign_weight = total_count / (benign_count) if benign_count > 0 else 1.0 
    malignant_weight = total_count / (malignant_count) if malignant_count > 0 else 1.0
    weights = torch.tensor([benign_weight, malignant_weight], device=DEVICE, dtype=torch.float32)
    # weights = weights / weights.sum()  # Normalize the weights
    
    # oversample the minority class, copy the sample to balance the classes
    minority_df = train_annotations[train_annotations['is_benign'] == 1]
    majority_df = train_annotations[train_annotations['is_benign'] == 0]
    minority_oversampled = minority_df.sample(n=len(majority_df), replace=True, random_state=config.random_state)
    
    train_annotations_oversampled = pd.concat([majority_df, minority_oversampled], ignore_index=True) 
    logger.info(f"Train annotations NOT oversampled: {len(train_annotations)} samples, {len(train_annotations[train_annotations['is_benign'] == 1])} benign and {len(train_annotations[train_annotations['is_benign'] == 0])} malignant")
    logger.info(f"Train annotations oversampled: {len(train_annotations_oversampled)} samples, {len(train_annotations_oversampled[train_annotations_oversampled['is_benign'] == 1])} benign and {len(train_annotations_oversampled[train_annotations_oversampled['is_benign'] == 0])} malignant")
    
    # compute difficulty
    # Columns: x1, y1, x2, y2, hu_mean
    def compute_difficulty(df):
        df["area"] = (df["bbox_x2"] - df["bbox_x1"]) * (df["bbox_y2"] - df["bbox_y1"])
        hu_min  = -1000
        hu_max = 500

        # Normalize HU and area between 0 and 1
        df["hu_norm"] = (df['nodule_mean_intensity'] - hu_min) / (hu_max - hu_min)
        df["area_norm"] = (df["area"] - df["area"].min()) / (df["area"].max() - df["area"].min())

        # Define difficulty as inverse of ease (higher is harder)
        df["difficulty"] = 1 - 0.5 * (df["hu_norm"] + df["area_norm"])
        
        # normalize difficulty to [0, 1]
        df["difficulty"] = (df["difficulty"] - df["difficulty"].min()) / (df["difficulty"].max() - df["difficulty"].min())
        return df
    
    train_annotations_oversampled = compute_difficulty(train_annotations_oversampled)    
    
    # Sampler to handle class imbalance AND curriculum learning, what a move boi
    sampler = CurriculumBalancedSampler(
        labels=train_annotations_oversampled['is_benign'].tolist(),
        difficulties=train_annotations_oversampled['difficulty'].tolist(),
        total_epochs=config.epochs,
        samples_per_epoch=config.batch_size * 600
    )
    
    # Create datasets
    # train_dataset = DLCSNoduleClassificationDataset(train_annotations, min_size=64, augment=config.augment, zoom_factor=0.8)
    train_dataset = DLCSNoduleClassificationDataset(train_annotations_oversampled,
                                                    min_size=64, 
                                                    augment=config.augment, 
                                                    zoom_factor=0.8, 
                                                    simulate_3ch=True,
                                                    square_patches=square_patches)
    
    val_dataset = DLCSNoduleClassificationDataset(val_annotations, 
                                                  min_size=64, 
                                                  augment=False, 
                                                  zoom_factor=0.8, 
                                                  simulate_3ch=True,
                                                  square_patches=square_patches)
    
    
    # Create data loaders
    train_loader = train_dataset.get_loader(batch_size=config.batch_size, shuffle=True, num_workers=config.dl_workers)
    val_loader = val_dataset.get_loader(batch_size=config.batch_size, shuffle=False, num_workers=config.dl_workers)

    model = model_class(num_classes=1) 
    
    model.to(DEVICE)
    if config.use_wandb:
        wandb.init(project="nodule-classification",
                   config=config,
                   name=f"{model.__class__.__name__} 3ch - {'square' if square_patches else 'original AR'} patches - F1",
                   notes="oversampled, augmented, zoom_factor=0.8, min_size=64, focal loss")
        wandb.watch(model, log="all", log_freq=300)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0.00001)
    # criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean').to(DEVICE) 
    # criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)  # Use BCEWithLogitsLoss for binary classification
    logger.info(f"Model initialized: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    scaler = GradScaler(enabled=config.amp)
    
    best_f1_avg = -1.0
    patience_counter = 0
    # Training loop
    for epoch in range(config.epochs):
        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch + 1} with best validation accuracy: {best_f1_avg:.4f}")
            break
        
        epoch_loss, epoch_acc = train_epoch(model, optimizer, train_loader, criterion, scaler)
        val_loss, val_acc, f1_avg = validation_epoch(model, val_loader, criterion)        
        scheduler.step()

        metrics = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "f1_score_avg": f1_avg
        }
        
        if f1_avg > best_f1_avg:
            best_f1_avg = f1_avg
            save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, name="best", metrics=metrics)
            logger.info(f"New best model saved at epoch {epoch + 1} with F1 score: {f1_avg:.4f}")
            patience_counter = 0
        else:   
            patience_counter += 1
        save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, name="last", metrics=metrics)

        if config.use_wandb:
            wandb.log(metrics)
    
    # load best model for evaluation
    best_checkpoint_path = Path(config.checkpoint_dir) / "nodule_classification" / model.__class__.__name__ / model.__class__.__name__ /f"checkpoint_best.pt"
    model = utils.load_model(best_checkpoint_path, model_class=model_class, device=DEVICE, num_classes=1)
    val_loss, val_acc = validation_epoch(model, val_loader, criterion, upload_cm=True)
    logger.info(f"Best model validation accuracy: {val_acc:.4f} with loss: {val_loss:.4f}")
    
    if config.use_wandb:
        wandb.finish()
        
    
if __name__ == "__main__":
    main()
