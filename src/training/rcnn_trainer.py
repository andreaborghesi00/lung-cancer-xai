import torch
import logging
from utils.utils import setup_logging
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List, Union
from torchmetrics.regression import MeanSquaredError
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import wandb
from config.config import get_config
from utils.metrics import iou_loss

class RCNNTrainer():
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
        self.use_wandb = use_wandb # wandb logging only during training
        
        self.train_metrics = self._init_metrics()
        self.val_metrics = self._init_metrics()
        if self.use_wandb:
            wandb.init(project=self.config.experiment_name)
            wandb.watch(self.model)
        
        self.logger.info("Initialized ROITrainer")
    
    def _init_metrics(self) -> Dict[str, Any]:
        return {
            "iou": MeanAveragePrecision(box_format='xyxy').to(self.device), # default format is xyxy, but annotations come in xywh
            # "mse": MeanSquaredError().to(self.device)
        }
    
    def _reset_metrics(self, metrics: Dict[str, Any]):
        for metric in metrics.values():
            metric.reset()
    
    def _format_box_for_map(self, boxes: List[Dict[str, torch.Tensor]], is_prediction: bool = False) -> List[Dict[str, torch.Tensor]]:
        """
        Format the bounding boxes for the MeanAveragePrecision metric (torchmetrics)
        """
        
        formatted_boxes = []
        for box_dict in boxes:
            formatted_dict = {
                "boxes": box_dict["boxes"].to(torch.float32)
            }
            
            if is_prediction: # one-liners are useful only if they do not hinder readability
                formatted_dict["scores"] = box_dict["scores"].to(torch.float32)
                formatted_dict["labels"] = box_dict["labels"].to(torch.int64)
            else:
                formatted_dict["scores"] = torch.ones(len(box_dict["boxes"]), dtype=torch.float32).to(self.device)
                formatted_dict["labels"] = torch.ones(len(box_dict["boxes"]), dtype=torch.int64).to(self.device)
            
            formatted_boxes.append(formatted_dict)
            
        return formatted_boxes
    
    # def _format_rcnn_output_for_map(self, output: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        
    
                    
    def save_checkpoint(self, epoch: Union[int, str], metrics: Dict[str, float]):
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

        
    def train_epoch(self, dl: DataLoader) -> Dict[str, float]:
        """
        Training epoch
        """
        self.model.train()
        self._reset_metrics(self.train_metrics)
        
        total_loss = 0.0
        pbar = tqdm(dl, desc="Training")
        
        for (images, targets) in pbar:
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets) # the model internally processes the loss
            
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            
            if self.use_wandb:
                wandb.log({"train_loss": losses.item()})
            
            self.optimizer.step()
            total_loss += losses.item()
            
            loss_dict["total_loss"] = losses
            loss_dict = {k: v.item() for k, v in loss_dict.items()}
            pbar.set_postfix(loss_dict)
        
        metrics = {
            "rcnn_avg_loss": total_loss / len(dl) # average loss
        }
        
        return metrics


    def validation(self, dl: DataLoader) -> Dict[str, float]:
        """
        Validation loop, uses mAP as metric, but can be extended to include other metrics
        """
        self.model.eval()
        self._reset_metrics(self.val_metrics)
        pbar = tqdm(dl, desc="Validation")
        
        with torch.no_grad():
            for images, targets in pbar:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                formatted_preds = self.model(images) # in eval mode, the model returns the output directly
                
                formatted_preds = self._format_box_for_map(formatted_preds, is_prediction=True)
                targets = self._format_box_for_map(targets)
    
                try:
                    self.val_metrics["iou"].update(formatted_preds, targets)
                except Exception as e:
                    self.logger.error(f"Error updating metrics: {str(e)}")
                    self.logger.debug(f"Predictions shape: {[p['boxes'].shape for p in formatted_preds]}")
                    self.logger.debug(f"Targets shape: {[t['boxes'].shape for t in targets]}")
                    raise
                
                pbar.set_postfix({"IoU": self.val_metrics["iou"].compute()['map'].item()})
        
                
        metrics = {
            "iou": self.val_metrics["iou"].compute()['map'].item()
        }
        
        return metrics

    
    def train(self, num_epochs: int):
        """
        Training loop
        """
        self.logger.info(f"Training for {num_epochs} epochs")
        best_val_loss = 0.0
        
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
            
            if val_metrics["iou"] > best_val_loss:
                best_val_loss = val_metrics["iou"]
                self.save_checkpoint(epoch+1, metrics) # the +1 is just to be consistent with the logs, epochs start at 1
            # save last model
            self.save_checkpoint("last", metrics)
                