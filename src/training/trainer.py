import torch
import logging
from utils.utils import setup_logging
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List
from torchmetrics.regression import MeanSquaredError
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import wandb
from config.config import get_config
from utils.metrics import iou_loss
class ROITrainer():
    def __init__(self,
                 model: nn.Module, 
                 optimizer: optim.Optimizer, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 checkpoint_dir: Optional[str] = None,
                 use_wandb: bool = False
                 ):
            
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging()
        self.config = get_config()
                        
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.use_wandb = use_wandb
        
        self.train_metrics = self._init_metrics()
        self.val_metrics = self._init_metrics()
        
        if self.use_wandb:
            wandb.init(project=self.config.experiment_name)
            wandb.watch(self.model)
        
        self.logger.info("Initialized ROITrainer")
        
    def _init_metrics(self) -> Dict[str, Any]:
        return {
            "iou": MeanAveragePrecision(box_format='xywh').to(self.device), # default format is xyxy, but annotations come in xywh
            # "mse": MeanSquaredError().to(self.device)
        }
        
    def _reset_metrics(self, metrics: Dict[str, Any]):
        for metric in metrics.values():
            metric.reset()
    
    def _format_box_for_map(self, boxes: torch.Tensor, scores: Optional[torch.Tensor] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Format the bounding boxes for the MeanAveragePrecision metric (torchmetrics)
        """
        
        if scores is None:
            scores = torch.ones(len(boxes)).to(self.device)
        return [
            {
                "boxes": boxes.to(torch.float32),
                "scores": scores.to(torch.float32),
                "labels": torch.zeros(len(boxes), dtype=torch.int64).to(self.device) # the box identifies only the class "tumors" hence the labels are all 0
            }
        ]
    
    def train_epoch(self, dl: DataLoader) -> Dict[str, float]:
        """
        Training epoch
        """
        self.model.train()
        self._reset_metrics(self.train_metrics)
        
        pbar = tqdm(self.train_loader, desc="Training")
        total_loss = 0.0
        for (images, targets) in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # loss = self.train_metrics["mse"](outputs, targets)
            loss = iou_loss(outputs, targets)
            if self.use_wandb:
                wandb.log({"train_loss": loss.item()})
            
            pred_boxes = self._format_box_for_map(outputs)
            target_boxes = self._format_box_for_map(targets)
            self.train_metrics["iou"].update(pred_boxes, target_boxes)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss (IoU)": loss.item()})
        
        metrics = {
            "iou": self.train_metrics["iou"].compute()['map'].item(),
            # "mse": self.train_metrics["mse"].compute().item()
            "homemade_iou": total_loss / len(dl) # average loss
        }
        
        return metrics
            

    def validation(self, dl: DataLoader) -> Dict[str, float]:
        """
        Validation loop
        """
        self.model.eval()
        self._reset_metrics(self.val_metrics)
        total_loss = 0.0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                # self.val_metrics["mse"](outputs, targets)
                loss = iou_loss(outputs, targets)
                
                pred_boxes = self._format_box_for_map(outputs)
                target_boxes = self._format_box_for_map(targets)
                self.val_metrics["iou"].update(pred_boxes, target_boxes)

                total_loss += loss.item()
                
        metrics = {
            "iou": self.val_metrics["iou"].compute()['map'].item(),
            # "mse": self.val_metrics["mse"].compute().item()
            "homemade_iou": total_loss / len(dl) # average loss
        }
        
        return metrics
                    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """
        Save the model checkpoint
        """
        if self.checkpoint_dir is None:
            return 
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")
        
        
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load the model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(f"Checkpoint loaded from {checkpoint_path} at epoch {checkpoint['epoch']}\n Metrics: {checkpoint['metrics']}")
        
        return checkpoint["epoch"], checkpoint["metrics"]
    
    def train(self, num_epochs: int):
        """
        Training loop
        """
        self.logger.info(f"Training for {num_epochs} epochs")
        best_val_iou = 0.0
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            train_metrics = self.train_epoch(self.train_loader)
            val_metrics = self.validation(self.val_loader)
            
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['iou']) # some schedulers may prefer to be called within the batch loop, but for now we'll call it at the end of the epoch
            
            metrics = {**train_metrics, **val_metrics} # merge the two dictionaries
            self.logger.info(f"Metrics: {metrics}")
            
            if self.use_wandb:
                wandb.log(metrics)
            
            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                self.save_checkpoint(epoch, metrics)
                