import torch
import logging
from utils.utils import setup_logging
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List, Union, Tuple
from torchmetrics.regression import MeanSquaredError
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import wandb
from config.config import get_config
from utils.metrics import iou_loss
import numpy as np
class RCNNTrainer():
    def __init__(self,
                 model: nn.Module, 
                 optimizer: optim.Optimizer = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 checkpoint_dir: Optional[str] = None,
                 use_wandb: bool = False
                 ):
            
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging(level=logging.DEBUG)
        self.config = get_config()
                        
        self.device = device

        self.model = model
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module  # Unwrap model for correct access
        self.model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.use_wandb = use_wandb # wandb logging only during training
        
        self.train_metrics = self._init_metrics()
        self.val_metrics = self._init_metrics()
        if self.use_wandb:
            wandb.init(project=self.config.project_name,
                       name=self.config.experiment_name,
                       notes=self.config.notes,
                       tags=[self.model.__class__.__name__,
                             f"{self.config.train_split_ratio} train split",
                             f"{self.config.val_test_split_ratio} val split",
                             self.optimizer.__class__.__name__,
                             self.scheduler.__class__.__name__ if self.scheduler is not None else "No scheduler",
                             "Pretrained" if self.model.weights is not None else "Not pretrained"])
            wandb.watch(self.model)
            # dump config to wandb
            wandb.config.update(self.config.__dict__)
        self.logger.info("Initialized RCNN Trainer")
    
    def _init_metrics(self) -> Dict[str, Any]:
        return {
            "mAP@5:95": MeanAveragePrecision(iou_thresholds=np.arange(0.05, 1.0, 0.05).tolist()).to(self.device), 
            "mAP@50": MeanAveragePrecision(iou_thresholds=[0.5], box_format='xyxy').to(self.device),
            "mAP@75": MeanAveragePrecision(iou_thresholds=[0.75], box_format='xyxy').to(self.device),
            "mAP@50:95": MeanAveragePrecision(iou_thresholds=np.arange(0.5, 1.0, 0.05).tolist(), box_format='xyxy').to(self.device),
            # "mse": MeanSquaredError().to(self.device)
        }
    
    def _reset_metrics(self, metrics: Dict[str, Any]):
        for metric in metrics.values():
            metric.reset()
    
    def _format_box_for_map(self, preds: List[Dict[str, torch.Tensor]], is_prediction: bool = False) -> List[Dict[str, torch.Tensor]]:
        """
        Format the bounding boxes for the MeanAveragePrecision metric (torchmetrics)
        """
        
        formatted_preds = []
        for pred_dict in preds:
            formatted_dict = {
                "boxes": pred_dict["boxes"].to(torch.float32)
            }
            
            if is_prediction: # one-liners are useful only if they do not hinder readability
                formatted_dict["scores"] = pred_dict["scores"].to(torch.float32)
                formatted_dict["labels"] = pred_dict["labels"].to(torch.int64)
            else:
                formatted_dict["scores"] = torch.ones(len(pred_dict["boxes"]), dtype=torch.float32).to(self.device)
                formatted_dict["labels"] = torch.ones(len(pred_dict["boxes"]), dtype=torch.int64).to(self.device)
            
            formatted_preds.append(formatted_dict)
            
        return formatted_preds
    
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
        
        checkpoint_path = self.checkpoint_dir / self.model.__class__.__name__ / f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
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
                    # self.val_metrics["mAP@5:95"].update(formatted_preds, targets)
                    for keys in self.val_metrics.keys():
                        self.val_metrics[keys].update(formatted_preds, targets)
                except Exception as e:
                    self.logger.error(f"Error updating metrics: {str(e)}")
                    self.logger.debug(f"Predictions shape: {[p['boxes'].shape for p in formatted_preds]}")
                    self.logger.debug(f"Targets shape: {[t['boxes'].shape for t in targets]}")
                    raise
                
                pbar.set_postfix({"mAP@5:95": self.val_metrics["mAP@5:95"].compute()['map'].item()})

        metrics = {k: v.compute()['map'].item() for k, v in self.val_metrics.items()}
        
        return metrics

    
    def validation_outlier_detection(self, dl: DataLoader) -> Dict[str, float]:
        self.model.eval()
        self._reset_metrics(self.val_metrics)
        pbar = tqdm(dl, desc="Validation Tomography Aware")
        
        with torch.no_grad():
            for tomographies, targets in pbar: # batch
                tomographies = torch.stack(tomographies) # B, D, C, H, W, although it would be necessary to do tomographies[0] since it is for some reason wrapped into a tuple, we would lose the B dimension, this is a way to both unwrap it and add the B dimension
                tomographies = tomographies.to(torch.device("cuda:0"))
                targets = targets[0]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets] 
                
                for batch_idx in range(tomographies.shape[0]): # tomography
                    tomography = tomographies[batch_idx, ...]
                    box_scores = []
                    preds_list = []
                    for slice_idx in range(tomography.shape[0]): # slice
                        image = tomography[slice_idx, ...]
                        image = image.unsqueeze(0)
                        preds = self.model(image)
                        
                        preds = self._format_box_for_map(preds, is_prediction=True) # only one prediction since we are using a single slice
                        preds_list.append(preds)
                        # list of (box, score) tuples
                        box_scores.extend([(preds[0]["boxes"][i], preds[0]["scores"][i]) for i in range(len(preds[0]['boxes']))])
                        
                    if box_scores is not None and len(box_scores) > 0:
                        # remove outlier bboxes from prediction
                        box_scores.sort(key=lambda x: x[1], reverse=True)
                        keep, skip = self._remove_outliers([box for box, _ in box_scores])
                        
                        # get the mean bbox
                        mean_bbox = self._get_mean_bbox([box_score[0] for i, box_score in enumerate(box_scores) if i in keep])
                        formatted_mean_bbox = {
                            "boxes": mean_bbox.view(1, 4),
                            "labels": torch.ones(1, dtype=torch.int64).to(self.device),
                            "scores": torch.ones(1, dtype=torch.float32).to(self.device)
                        }
                    else:
                        formatted_mean_bbox = self._empty_prediction()
                    
                    targets = [self._format_box_for_map([t], is_prediction=False) for t in targets]
                    
                    # update the metrics
                    try:
                        for keys in self.val_metrics.keys():
                            for i in range(tomography.shape[0]): # we need to update the metrics for each slice to be comparable with the other validation loop
                                if i in skip:
                                    self.val_metrics[keys].update([formatted_mean_bbox], targets[i])
                                else:
                                    self.val_metrics[keys].update(preds_list[i], targets[i])
                    except Exception as e:
                        self.logger.error(f"Error updating metrics: {str(e)}")
                        self.logger.info(f"Predictions shape: {[p['boxes'].shape for p in preds]}")
                        self.logger.info(f"Targets shape: {[t['boxes'].shape for t in targets]}")
                        raise
                    
                pbar.set_postfix({"mAP@5:95": self.val_metrics["mAP@5:95"].compute()['map'].item()})
                
        metrics = {k: v.compute()['map'].item() for k, v in self.val_metrics.items()}
        return metrics
                                            
                        
        
    def _empty_prediction(self) -> Dict[str, torch.Tensor]:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32).to(self.device),
            "labels": torch.empty(0, dtype=torch.int64).to(self.device),
            "scores": torch.empty(0, dtype=torch.float32).to(self.device)
        }
        
        
    def _remove_outliers(self, boxes: List[torch.Tensor], threshold: float = 0.1) -> Tuple[List[int], List[int]]:
        """
        Given a list of boxes, remove the outlying boxes based on the distribution of the boxes.
        given the IQR, we can remove the boxes that are outside the 1.5 * IQR range from the median for each dimension      
        
        returns the list of indices of the kept boxes and the removed boxes
        """
        if len(boxes) == 0:
            return boxes
        
        original_count = len(boxes)
        boxes = torch.stack(boxes)
        kept_boxes_idx = []
        removed_boxes_idx = []
        multipliers = [1.5, 2.0, 2.5, 3.0]  # progressively less aggressive filtering
        
        for multiplier in multipliers:
            q1 = torch.quantile(boxes, 0.25, dim=0)
            q3 = torch.quantile(boxes, 0.75, dim=0)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            kept_boxes_idx = []
            removed_boxes_idx = []
            for i in range(len(boxes)):
                if torch.all(boxes[i] > lower_bound) and torch.all(boxes[i] < upper_bound):
                    kept_boxes_idx.append(i)
                else:
                    removed_boxes_idx.append(i)
            
            # if we have a reasonable number of boxes, stop
            if len(kept_boxes_idx) > 0 and len(kept_boxes_idx) >= min(2, original_count * 0.3):
                self.logger.debug(f"Used IQR multiplier {multiplier}: kept {len(kept_boxes_idx)}/{original_count} boxes")
                break
        
        if len(kept_boxes_idx) == 0 and original_count > 0:
            self.logger.warning("All boxes were filtered out as outliers, using top confidence boxes instead")
            self.logger.warning(f"Original count: {original_count}")
            # Fall back to using the top N most confident boxes
            # Sort boxes by confidence if available, otherwise use the first few
            return [0], np.arange(1, len(boxes))
        return kept_boxes_idx, removed_boxes_idx
    
    
    def _get_mean_bbox(self, boxes: List[torch.Tensor]) -> torch.Tensor:
        """
        Given a list of bboxes, it returns a single bbox that is the mean of all bboxes
        """ 
        if len(boxes) == 0:
            return boxes
        
        boxes = torch.stack(boxes)
        mean_bbox = torch.mean(boxes, dim=0)
        
        if mean_bbox.shape != torch.Size([4]):
            self.logger.warning(f"Mean bbox has unexpected shape: {mean_bbox.shape}, reshaping to [4]")
            mean_bbox = mean_bbox.view(4)
        
        return mean_bbox
        
        
    def train(self, train_loader, val_loader, num_epochs: int, patience: int = 5):
        """
        Training loop
        """
        self.logger.info(f"Training for {num_epochs} epochs")
        best_val_loss = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            train_metrics = self.train_epoch(train_loader) 
            val_metrics = self.validation(val_loader)
            
            if self.scheduler is not None:
                # self.scheduler.step(val_metrics['mAP@5:95']) # some schedulers may prefer to be called within the batch loop, but for now we'll call it at the end of the epoch
                self.scheduler.step()
                # track LR
                if self.use_wandb:
                    wandb.log({"learning_rate": self.scheduler.get_last_lr()[0]})
                
            metrics = {**train_metrics, **val_metrics} # merge the two dictionaries
            self.logger.info(f"Metrics: {metrics}")
            
            if self.use_wandb:
                wandb.log(metrics)
            
            if val_metrics["mAP@5:95"] > best_val_loss:
                best_val_loss = val_metrics["mAP@5:95"]
                self.save_checkpoint("best", metrics)
                patience_counter = 0
            else:
                patience_counter += 1
            # save last model
            self.save_checkpoint("last", metrics)
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        wandb.finish()
        self.logger.info("Training complete")
                