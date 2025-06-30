import torch
import supervision as sv
import os
import transformers
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import json
from PIL import Image
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers.image_utils import AnnotationFormat
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pytorch_lightning import Trainer
import wandb

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "facebook/detr-resnet-50"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

DATA_PATH = Path("/gwdata/users/aborghesi/DLCS/2d_1_to_6_resampled/")
ANNOTATION_PATH = Path("/gwdata/users/aborghesi/DLCS/2d_annotations.csv")

ANNOTATIONS_COCO_PATH = ANNOTATION_PATH.parent / "annotations_coco_full.json"
ANNOTATIONS_COCO_TRAIN_PATH = ANNOTATION_PATH.parent / "annotations_coco_train.json"
ANNOTATIONS_COCO_VAL_PATH = ANNOTATION_PATH.parent / "annotations_coco_val.json"
ANNOTATIONS_COCO_SMALL_PATH = ANNOTATION_PATH.parent / "annotations_coco_small.json"

TRAIN_SPLIT = 0.7
VAL_TEST_SPLIT = 0.5
RANDOM_STATE = 123

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    pixel_values = torch.stack(pixel_values, dim=0).to(torch.float32)
    
    labels = [item[1] for item in batch]
    annotations = [item[2] for item in batch] if len(batch[0]) >= 3 else None
    pixel_mask = [item[3] for item in batch] if len(batch[0]) >= 4 else None
    pixel_mask = torch.stack(pixel_mask, dim=0).to(torch.uint8) if pixel_mask is not None else None
    orig_size = [item[4] for item in batch] if len(batch[0]) >= 5 else None
    image_ids = [item[5] for item in batch] if len(batch[0]) >= 6 else None

    return {
        'pixel_values': pixel_values,
        'annotations': annotations,
        'labels': labels,
        'pixel_mask': pixel_mask,
        'orig_size': orig_size,
        'image_id': image_ids
    }

class DlcsCoco(CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        annotation_path: str,
        image_processor, 
        train: bool = True,
        max_hu=500,
        min_hu=-1000
    ):
        super(DlcsCoco, self).__init__(image_directory_path, annotation_path)
        self.image_processor = image_processor
        self.max_hu = max_hu
        self.min_hu = min_hu

        
    def clip_normalize_hu(self, image_array):
        image_array = np.clip(image_array, self.min_hu, self.max_hu)
        return (image_array - self.min_hu) / (self.max_hu - self.min_hu)


    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        # print(f"Loading image from path: {path}")
        npy_path = Path(self.root) / path
        
        array = np.load(npy_path)
        array = self.clip_normalize_hu(array)
        
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        
        return Image.fromarray(array).convert("RGB")


    def __getitem__(self, idx):
        images, annotations = super(DlcsCoco, self).__getitem__(idx)
        orig_size = images.size
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        # print(f"Encoding keys: {encoding.keys()}")
        # print(f"Pixel values shape: {encoding['pixel_values'].shape}")
        pixel_values = encoding["pixel_values"].squeeze()
        pixel_mask = encoding["pixel_mask"].squeeze()
        # print(f"Pixel values shape: {encoding['pixel_values'].shape}")
        
        target = encoding["labels"][0]

        return pixel_values, target, annotations, pixel_mask, orig_size, image_id

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, train_loader, val_loader, val_annotations_json, image_processor):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT, 
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
        # for n,p in self.model.named_parameters():
        #     if "backbone" in n or "transformer.encoder" in n:
        #         p.requires_grad = False
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.coco_gt = COCO(val_annotations_json)
        self._predictions = []
        
        self.image_processor = image_processor
        
        wandb.init(project="dev_roi",
                       name="Detr",)
        wandb.watch(self.model)

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        # print(loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        wandb.log(loss_dict)
        
        return loss

    # def validation_step(self, batch, batch_idx):
    #     loss, loss_dict = self.common_step(batch, batch_idx)     
    #     self.log("validation/loss", loss)
    #     for k, v in loss_dict.items():
    #         self.log("validation_" + k, v.item())
            
    #     return loss

    def on_validation_epoch_start(self):
        # reset between epochs
        self._predictions = []

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask   = batch["pixel_mask"]
        outputs = self.model(pixel_values=pixel_values,
                             pixel_mask=pixel_mask)

        # print(f"Batch keys: {batch.keys()}")
        
        # post-process with HF’s image processor
        processed = self.image_processor.post_process_object_detection(
            outputs,
            threshold=0.2,
            target_sizes=[(h, w) for h, w in batch["orig_size"]]
        )

        # gather for COCOeval
        for img_id, det in zip(batch["image_id"], processed):
            for box, score, label in zip(det["boxes"],
                                         det["scores"],
                                         det["labels"]):
                x1, y1, x2, y2 = box.tolist()
                self._predictions.append({
                    "image_id":    int(img_id),
                    "category_id": int(label),
                    # COCO wants [x,y,width,height]
                    "bbox":        [x1, y1, x2 - x1, y2 - y1],
                    "score":       float(score)
                })

        # log your losses as before …
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("val/loss", loss)

    def on_validation_epoch_end(self):
        if len(self._predictions) == 0:
            print("No predictions made during validation step.")
            return
        
        # track raw IoU
        
        ious = []
        # group preds by image
        preds_by_image = {}
        for p in self._predictions:
            preds_by_image.setdefault(p["image_id"], []).append(p)

        for img_id, preds in preds_by_image.items():
            # pick the single highest‐score prediction
            best = max(preds, key=lambda x: x["score"])
            pred_box = best["bbox"]  # [x, y, w, h]

            # load GT (assumes 1 GT per image in your overfit test)
            ann_ids = self.coco_gt.getAnnIds(imgIds=[img_id])
            gts = self.coco_gt.loadAnns(ann_ids)
            if not gts:
                continue
            gt_box = gts[0]["bbox"]

            ious.append(self.box_iou(pred_box, gt_box))

        mean_iou = sum(ious) / len(ious) if ious else 0.0
        wandb.log({"val_mean_iou": mean_iou})
        
        # load detections
        coco_dt   = self.coco_gt.loadRes(self._predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # coco_eval.stats → [
        #   AP @[.5:.95], AP @.50, AP @.75, AP_small, AP_medium, AP_large,
        #   AR@1, AR@10, AR@100, AR_small, AR_medium, AR_large
        # ]
        stats = coco_eval.stats
        self.log("val_AP",    stats[0])
        self.log("val_AP50",  stats[1])
        self.log("val_AP75",  stats[2])

    def configure_optimizers(self):
        # head_params = [p for n,p in self.model.named_parameters() if p.requires_grad]

        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        # return torch.optim.AdamW(head_params, lr=self.lr, weight_decay=self.weight_decay)
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    
    @staticmethod
    def box_iou(box1, box2):
        # box format: [x, y, w, h]
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x1_max = max(x1, x2)
        y1_max = max(y1, y2)
        x2_min = min(x1 + w1, x2 + w2)
        y2_min = min(y1 + h1, y2 + h2)

        inter_w = max(0, x2_min - x1_max)
        inter_h = max(0, y2_min - y1_max)
        inter_area = inter_w * inter_h

        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
def main():
    annotations = pd.read_csv(ANNOTATION_PATH)
    # read the annotations

    with open(ANNOTATIONS_COCO_PATH, "r") as f:
        full_json_annotations = json.load(f)
    with open(ANNOTATIONS_COCO_TRAIN_PATH, "r") as f:
        train_json_annotations = json.load(f)
    with open(ANNOTATIONS_COCO_VAL_PATH, "r") as f:
        val_json_annotations = json.load(f)
        
    transform = DetrImageProcessor.from_pretrained(CHECKPOINT)
    transform.do_rescale = False
    transform.do_resize = False
    transform.size = {"shortest_edge": 800, "longest_edge": 1333}
    transform.do_pad = True
    transform.pad_size = {"height": 800, "width": 800}
    # transform.pad_size = None  # Let the processor handle padding dynamically
    transform.do_normalize = False
    transform.format = AnnotationFormat.COCO_DETECTION
    
    
    train_ds = DlcsCoco(
    image_directory_path=DATA_PATH,
    annotation_path=ANNOTATIONS_COCO_TRAIN_PATH,
    image_processor=transform
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_ds = DlcsCoco(
        image_directory_path=DATA_PATH,
        annotation_path=ANNOTATIONS_COCO_VAL_PATH,
        image_processor=transform
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    small_ds = DlcsCoco(
        image_directory_path=DATA_PATH,
        annotation_path=ANNOTATIONS_COCO_SMALL_PATH,
        image_processor=transform
    )
    
    small_dl = DataLoader(
        small_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        # num_workers=1,
        # pin_memory=True,
    )
    
    model = Detr(
        lr=1e-4, 
        # lr=1e-2, 
        lr_backbone=1e-5, 
        weight_decay=1e-4,
        # weight_decay=0,
        train_loader=small_dl,
        val_loader=small_dl,
        val_annotations_json=ANNOTATIONS_COCO_SMALL_PATH,
        image_processor=transform
    )
    
    trainer = Trainer(devices=1,
                      accelerator="gpu", 
                      max_epochs=500, 
                      log_every_n_steps=1, 
                      enable_progress_bar=True, 
                      gradient_clip_val=None,
                      check_val_every_n_epoch=50,
                    #   overfit_batches=1
    )
    
    trainer.fit(model=model)


if __name__ == "__main__":
    main()