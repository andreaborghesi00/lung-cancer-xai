import torch
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.nodule_classification_dataset import DLCSNoduleClassificationDataset
from models.nodule_classifiers import NoduleClassifier, NoduleClassifierMobileNet, NoduleClassifierEfficientNetv2S
from config.config_2d import get_config
from tqdm import tqdm
import wandb

def main():
    config = get_config()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
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
    model = NoduleClassifierEfficientNetv2S(num_classes=2)
    model.to(device)
    
    wandb.init(project="nodule-classification", config=config, name=f"{model.__class__.__name__}_run")
    wandb.watch(model, log="all", log_freq=300)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    logger.info(f"Model initialized: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    pbar = tqdm(train_loader, desc="Training Progress", unit="batch")
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({"loss": loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        logger.info(f"Epoch [{epoch+1}/{config.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        scheduler.step()
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        
    
    
if __name__ == "__main__":
    main()
