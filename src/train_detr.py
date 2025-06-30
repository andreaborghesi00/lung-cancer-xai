import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from tqdm import tqdm
import random
import logging
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json # For creating COCO format json temporarily
import tempfile 
import numpy as np
import matplotlib.pyplot as plt

# Import from your detr_data.py
from data.detr_datasets import (
    load_and_prepare_annotations_from_csv,
    CTNoduleDataset,
    collate_fn, # We need IMAGE_PROCESSOR for this, so initialize it here too or pass it
    train_transforms,
    val_transforms,
    CLASS_TO_ID,
    NUM_CLASSES,
    MODEL_CHECKPOINT,
    IMAGE_PROCESSOR
    # CSV configurations are used within load_and_prepare_annotations_from_csv
)
from config.config_2d import get_config # Assuming your config is here

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration from detr_data and config.py ---
config = get_config() # From your config file
config.validate()

# Training params (can also be moved to config.py)
LEARNING_RATE = config.learning_rate  # Example: Get from your config
BATCH_SIZE = config.batch_size        # Example
NUM_EPOCHS = config.epochs        # Example
OUTPUT_DIR = Path("detr_output/")      # Example: "./detr_model_output"
DEVICE = torch.device(config.device if torch.cuda.is_available() else "cpu") 
# Initialize Image Processor (needs to be the same instance used in collate_fn)
# If collate_fn is in detr_data.py and uses a global IMAGE_PROCESSOR, ensure it's initialized
# For clarity, let's define it here and pass it or make collate_fn more flexible.

# Modify collate_fn to accept image_processor (recommended for clarity)
# In detr_data.py:
# def collate_fn(batch, image_proc):
#     pixel_values = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
#     batch_processed = image_proc(images=pixel_values, annotations=labels, return_tensors="pt")
#     return batch_processed
# Then in DataLoader below:
# train_dataloader = DataLoader(..., collate_fn=lambda b: collate_fn(b, image_processor))

# For now, assuming collate_fn in detr_data.py uses its globally defined IMAGE_PROCESSOR
# Ensure that global one is initialized when detr_data is imported.
# The current detr_data.py does this: IMAGE_PROCESSOR = DetrImageProcessor.from_pretrained(MODEL_CHECKPOINT)
# So, we can directly use the imported collate_fn.


def get_detr_model(num_classes, checkpoint=MODEL_CHECKPOINT, learning_rate=1e-5, weight_decay=1e-4):
    """
    Initializes the DETR model and optimizer.
    """
    # Load pre-trained model configuration
    # Use revision="no_timm" for official DETR resnet weights
    detr_config = DetrConfig.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        revision="no_timm", # Important for ResNet backbone from original DETR release
        # use_timm_backbone=False
    )
    logger.info(f"Loading DETR model from checkpoint: {checkpoint} with {num_classes} classes.")
    # Load pre-trained model, ignore_mismatched_sizes for the classification head
    model = DetrForObjectDetection.from_pretrained(
        checkpoint,
        config=detr_config,
        ignore_mismatched_sizes=True, # Allows changing num_labels
        revision="no_timm", # Important for ResNet backbone from original DETR release
        # use_timm_backbone=False
    )
    
    # Optimizer
    # Original DETR paper used different LRs for backbone vs transformer.
    # For simplicity, start with a single LR.
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": learning_rate,  # e.g., 1e-4 for transformer and heads
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": learning_rate * 0.1, # e.g., 1e-5 for backbone (add this to your config.py)
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    return model, optimizer


def train_one_epoch(model, data_loader, optimizer, device, epoch_num):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num} [Training]")

    for batch_idx, batch in enumerate(progress_bar):
        # Batch items are already processed by collate_fn using DetrImageProcessor
        # pixel_values: (batch_size, num_channels, height, width)
        # pixel_mask: (batch_size, height, width)
        # labels: list of dictionaries, each with 'class_labels', 'boxes', 'image_id', 'area', 'iscrowd'
        # These tensors in 'labels' also need to be on the device.

        pixel_values = batch["pixel_values"].to(device)
        # bboxes = batch["labels"][0]["boxes"].to(device) # Example for first item
        # bboxes = bboxes.cpu().numpy()
        # # Save the first batch of pixel values to visualize
        # sample_image = pixel_values[0].cpu().permute(1, 2, 0).numpy()
        # plt.imshow(sample_image)
        # plt.axis('off')
        # plt.gca().add_patch(plt.Rectangle(
        #     ((bboxes[0][0].item() - (bboxes[0][2]/2)) * 750, (bboxes[0][1].item() - (bboxes[0][3]/2)) * 750), bboxes[0][2].item() * 750, bboxes[0][3].item()* 750, edgecolor='red', facecolor='none'
        #     ))
        # plt.savefig(os.path.join(OUTPUT_DIR, f"processed_image_batch.png"))
        # plt.close()
        
        # Ensure labels are correctly moved to device
        # DetrImageProcessor usually returns labels where 'boxes' and 'class_labels' are tensors.
        labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch["labels"]]
        
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        optimizer.zero_grad()
        
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict 

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # max_norm
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "loss_ce": f"{loss_dict['loss_ce'].item():.4f}",
            "loss_bbox": f"{loss_dict['loss_bbox'].item():.4f}",
            "loss_giou": f"{loss_dict['loss_giou'].item():.4f}"
            })

    avg_epoch_loss = total_loss / len(data_loader)
    logger.info(f"Epoch {epoch_num} Training Average Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


@torch.no_grad()
def evaluate(model, data_loader, device, epoch_num, image_processor):
    model.eval()
    
    # For COCO evaluation
    coco_gt_list = [] 
    coco_pred_list = [] 

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num} [Validation]")

    for batch_idx, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        # original_labels are the ground truths from the dataloader BEFORE DetrImageProcessor formats them for loss
        # We need to get the original sizes and image_ids from the batch['labels']
        # The batch['labels'] from collate_fn are already formatted for loss.
        # We need the 'orig_size' and 'image_id' that DetrImageProcessor adds.
        
        # Let's assume your collate_fn ensures 'orig_size' and 'image_id' are present in batch['labels'] items
        # The 'image_id' in batch['labels'][i]['image_id'] should be the unique int ID.
        # The DetrImageProcessor adds 'orig_size' to each target dict when processing.

        # --- Store Ground Truths for COCO eval ---
        for i in range(len(batch["pixel_values"])):
            target_item = batch["labels"][i]
            img_id = int(target_item["image_id"].item()) # Ensure Python int
            gt_norm_boxes_padded_dim = target_item["boxes"].cpu() 
    
            # target_item["size"] is the padded H,W (e.g., 750, 750)
            # target_item["orig_size"] is the original image H,W before padding
            padded_h, padded_w = target_item["size"].cpu().numpy()
            # orig_h, orig_w = target_item["orig_size"].cpu().numpy() # We might not need this for GT unnorm if above is true

            for j in range(len(target_item["class_labels"])):
                norm_box_padded_dim = gt_norm_boxes_padded_dim[j] 
                
                # Unnormalize using the PADDED dimensions to get absolute coords on the padded canvas
                abs_center_x = norm_box_padded_dim[0] * padded_w
                abs_center_y = norm_box_padded_dim[1] * padded_h
                abs_width    = norm_box_padded_dim[2] * padded_w
                abs_height   = norm_box_padded_dim[3] * padded_h

                # Convert from [center_x, center_y, width, height] to COCO's [xmin, ymin, width, height]
                # These are now absolute pixel coordinates relative to the top-left of the padded canvas.
                # Since the original image content is also at the top-left of the padded canvas,
                # these are the correct absolute coordinates for COCO GT.
                abs_xmin = abs_center_x - (abs_width / 2.0)
                abs_ymin = abs_center_y - (abs_height / 2.0)
                
                current_area = float(abs_width * abs_height)

                gt_ann = {
                    "image_id": img_id,
                    "category_id": int(target_item["class_labels"][j].item()),
                    "bbox": [float(abs_xmin), float(abs_ymin), float(abs_width), float(abs_height)],
                    "area": current_area, # Recalculated area from absolute w,h
                    "iscrowd": 0,
                    "id": len(coco_gt_list) 
                }
                coco_gt_list.append(gt_ann)
        # --- Get Model Predictions ---
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        # outputs.logits: (batch_size, num_queries, num_classes + 1)
        # outputs.pred_boxes: (batch_size, num_queries, 4) in (center_x, center_y, width, height) normalized format

        # --- Post-process predictions ---
        original_sizes = torch.stack([label['orig_size'] for label in batch['labels']]).to(device)
        
        post_process_targets = [{"image_id": label["image_id"], "orig_size": label["orig_size"]} for label in batch["labels"]]

        results = image_processor.post_process_object_detection(
            outputs,
            threshold=0.0, # Low threshold to get all possible detections; COCOeval will handle score thresholds
            target_sizes=original_sizes
        )
        # `results` is a list of dicts, one per image in the batch.
        # Each dict has: 'scores' (tensor), 'labels' (tensor), 'boxes' (tensor, in [xmin, ymin, xmax, ymax] format)

        for i, res_dict in enumerate(results):
            img_id = int(post_process_targets[i]["image_id"].item()) # Ensure Python int
            scores = res_dict["scores"].cpu().numpy()
            labels = res_dict["labels"].cpu().numpy() # Array of numpy.int
            boxes_abs_xyxy = res_dict["boxes"].cpu().numpy() # Array of numpy.float

            for score_val, label_id_val, box_xyxy in zip(scores, labels, boxes_abs_xyxy):
                # box_xyxy is a numpy array [xmin, ymin, xmax, ymax]
                pred_bbox_xywh = [
                    float(box_xyxy[0]),
                    float(box_xyxy[1]),
                    float(box_xyxy[2] - box_xyxy[0]),
                    float(box_xyxy[3] - box_xyxy[1])
                ]
                if pred_bbox_xywh[2] < 0: pred_bbox_xywh[2] = 0.0
                if pred_bbox_xywh[3] < 0: pred_bbox_xywh[3] = 0.0
                
                pred_ann = {
                    "image_id": img_id,
                    "category_id": int(label_id_val), 
                    "bbox": pred_bbox_xywh, 
                    "score": float(score_val),
                }
                coco_pred_list.append(pred_ann)

    # --- Perform COCO Evaluation ---
    if not coco_gt_list or not coco_pred_list:
        logger.warning("No ground truths or predictions found for COCO evaluation. Skipping.")
        return None # Or return a dict of Nones/zeros

    # Create temporary JSON files for COCO API
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_gt_file, \
         tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_pred_file:
        
        # Prepare COCO ground truth structure
        # It needs 'images', 'annotations', 'categories'
        # 'images': [{"id": img_id, "height": h, "width": w}, ...]
        # 'annotations': the coco_gt_list
        # 'categories': [{"id": cat_id, "name": cat_name}, ...]
        
        # We need unique image_ids and their original dimensions for the 'images' field of coco_gt_json
        # This info might need to be collected more systematically during data loading or passed along
        # For now, let's try to infer from `batch['labels']` or `data_loader.dataset`
        # A robust way is to have `data_loader.dataset.get_image_info(image_id)`
        # For simplicity, let's assume we can reconstruct it or have a list of all image_ids and sizes.
        # The `batch["labels"]` already has `orig_size` and `image_id`.
        # We need to collect all unique `image_id` and their `orig_size`.

        if hasattr(data_loader.dataset, 'dataset_items'): # CTNoduleDatasetRevised has this
            # We need a way to map the integer image_id back to the original image to get its size
            # This is where having a direct mapping in the dataset class would be useful.
            # `CTNoduleDatasetRevised` gets (image_path, target_coco_format)
            # `target_coco_format` contains the integer `image_id`.
            # To get image H,W for the COCO 'images' field:
            # Iterate through val_dataset.dataset_items. For each item, load the image, get its original H,W
            # and store it against the item[1]['image_id'].
            # This is inefficient to do here. It's better if dataset_items stores this.

            # Hacky way for now: assume all val images were processed and orig_size is in batch['labels']
            # This is not robust as not all images might appear in the last batch.
            # For proper COCO format, the 'images' field should list all images evaluated.
            
            # Robust Solution: Modify `load_and_prepare_annotations_from_csv` to also store image dimensions.
            # Or, iterate through `val_dataset.dataset_items` and load each image to get dims.
            # Let's assume for now that `post_process_targets` (derived from `batch['labels']`) covers all image_ids and their orig_sizes.
            temp_image_infos_coco = []
            seen_img_ids_for_coco_images_field = set()
            for batch_val in data_loader: # Iterate again to get all orig_sizes (inefficient but works for now)
                 for label_dict_in_batch in batch_val["labels"]:
                    img_id_val = label_dict_in_batch["image_id"].item()
                    if img_id_val not in seen_img_ids_for_coco_images_field:
                        orig_size_val = label_dict_in_batch["orig_size"].cpu().numpy()
                        temp_image_infos_coco.append({"id": img_id_val, "height": int(orig_size_val[0]), "width": int(orig_size_val[1])})
                        seen_img_ids_for_coco_images_field.add(img_id_val)


        coco_categories = [{"id": class_id, "name": class_name, "supercategory": "nodule_super"} 
                           for class_name, class_id in CLASS_TO_ID.items()] # Use your CLASS_TO_ID

        coco_gt_json_data = {
            "images": temp_image_infos_coco, # List of {"id": int, "height": int, "width": int}
            "annotations": coco_gt_list,
            "categories": coco_categories
        }
        json.dump(coco_gt_json_data, tmp_gt_file)
        json.dump(coco_pred_list, tmp_pred_file) # Predictions format is just a list of annotation dicts
        
        tmp_gt_file.flush()
        tmp_pred_file.flush()

        coco_gt_api = COCO(tmp_gt_file.name)
        coco_dt_api = coco_gt_api.loadRes(tmp_pred_file.name) # Load predictions

        coco_eval = COCOeval(coco_gt_api, coco_dt_api, iouType='bbox')
        coco_eval.params.iouThrs = np.linspace(.05, .95, int(np.round((.95 - .05) / .05)) + 1, endpoint=True)
        # Set specific IoU thresholds for evaluation if desired (though COCOeval does many by default)
        # coco_eval.params.iouThrs = np.array([0.10, 0.50, 0.75]) # Example: just these three
        # The default coco_eval.params.iouThrs is np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        # which covers [0.5, 0.55, ..., 0.95]

        # To evaluate at a specific threshold like 0.1:
        # You can modify coco_eval.params.iouThrs before running evaluate() and summarize()
        # Or, run evaluate multiple times with different params.iouThrs.
        # However, COCOeval.summarize() prints a table with standard metrics including:
        #  - AP @ IoU=0.50:0.95 (primary challenge metric)
        #  - AP @ IoU=0.50
        #  - AP @ IoU=0.75
        #  - AR @ IoU=0.50:0.95 (across different maxDets)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize() # Prints the standard COCO metrics table

        # Extract specific metrics
        stats = coco_eval.stats # array of 12 metrics
        
        # Standard COCO Metrics:
        # stats[0]: AP @ IoU=0.50:0.95 | area=   all | maxDets=100
        # stats[1]: AP @ IoU=0.50      | area=   all | maxDets=100
        # stats[2]: AP @ IoU=0.75      | area=   all | maxDets=100
        # stats[3]: AP @ IoU=0.50:0.95 | area= small | maxDets=100
        # stats[4]: AP @ IoU=0.50:0.95 | area=medium | maxDets=100
        # stats[5]: AP @ IoU=0.50:0.95 | area= large | maxDets=100
        # stats[6]: AR @ IoU=0.50:0.95 | area=   all | maxDets=  1
        # stats[7]: AR @ IoU=0.50:0.95 | area=   all | maxDets= 10
        # stats[8]: AR @ IoU=0.50:0.95 | area=   all | maxDets=100  (This is often reported as mAR)
        # stats[9]: AR @ IoU=0.50:0.95 | area= small | maxDets=100
        # stats[10]: AR @ IoU=0.50:0.95 | area=medium | maxDets=100
        # stats[11]: AR @ IoU=0.50:0.95 | area= large | maxDets=100
        
        metrics_dict = {
            "AP_5_95": stats[0], # mAP
            "AP_50": stats[1],
            "AP_75": stats[2],
            "AR_max100_5_95": stats[6], # mAR
            # You can add more from `stats` if needed
        }
        logger.info(f"COCO Metrics: {metrics_dict}")

        # Clean up temporary files
        os.remove(tmp_gt_file.name)
        os.remove(tmp_pred_file.name)
        
        return metrics_dict # Or just stats[0] if you want to track primary mAP for best model saving

def main():
    logger.info(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global IMAGE_PROCESSOR
    # --- 1. Load and Prepare Annotations (from detr_data.py) ---
    logger.info("Loading and preparing annotations...")
    # Ensure CLASS_TO_ID is defined; it's imported from detr_data
    all_dataset_items = load_and_prepare_annotations_from_csv(config.annotation_path, CLASS_TO_ID)
    
    if not all_dataset_items:
        logger.error("No data loaded. Exiting.")
        return

    # --- 2. Split data into train and validation ---
    # Consider using sklearn.model_selection.train_test_split for more robust splitting,
    # especially if you want stratification by some criteria later.
    random.seed(config.seed if hasattr(config, 'seed') else 42) # Use seed from config if available
    random.shuffle(all_dataset_items)
    
    val_split_ratio = config.validation_split if hasattr(config, 'validation_split') else 0.1
    val_split_idx = int(val_split_ratio * len(all_dataset_items))
    
    if val_split_idx == 0 and len(all_dataset_items) > 1: # Ensure at least one val sample if possible and desired
        if len(all_dataset_items) > BATCH_SIZE : # only if we have enough for a batch
             val_split_idx = 1 
        else: # otherwise no validation
            logger.warning("Not enough samples for a validation batch, skipping validation.")
            val_split_idx = 0


    train_items = all_dataset_items[val_split_idx:]
    val_items = all_dataset_items[:val_split_idx]

    logger.info(f"Training samples: {len(train_items)}, Validation samples: {len(val_items)}")

    if not train_items:
        logger.error("Not enough data for training after split. Aborting.")
        return

    # --- 3. Create Datasets and DataLoaders ---
    train_dataset = CTNoduleDataset(dataset_items=train_items, transforms=train_transforms)
    
    # DataLoader parameters from config
    num_workers = config.num_workers if hasattr(config, 'num_workers') else 4
    pin_memory_flag = True if DEVICE.type == 'cuda' else False


    # Use the collate_fn from detr_data.py which uses the globally defined IMAGE_PROCESSOR there
    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=collate_fn, # from detr_data.py
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory_flag
    )

    val_dataloader = None
    if val_items:
        val_dataset = CTNoduleDataset(dataset_items=val_items, transforms=val_transforms) # Use val_transforms
        val_dataloader = DataLoader(
            val_dataset, 
            collate_fn=collate_fn, # from detr_data.py
            batch_size=BATCH_SIZE, 
            shuffle=False, # No need to shuffle validation data
            num_workers=num_workers,
            pin_memory=pin_memory_flag
        )
    else:
        logger.info("No validation data, skipping validation loader and evaluation during training.")

    # --- 4. Initialize Model and Optimizer ---
    model, optimizer = get_detr_model(
        num_classes=NUM_CLASSES, # from detr_data.py
        learning_rate=LEARNING_RATE,
        # weight_decay=WEIGHT_DECAY # keep default for now
    )
    model.to(DEVICE)

    logger.info("Starting training...")
    best_val_loss = float('inf')
    logger.info(f"Params being trained: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    train_loss = -1.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, DEVICE, epoch)
        
        val_loss_dict = None
        if val_dataloader:
            val_loss_dict = evaluate(model, val_dataloader, DEVICE, epoch, IMAGE_PROCESSOR   )
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss_dict}")
            
            if val_loss_dict['AP_5_95'] > best_val_loss:
                best_val_loss = val_loss_dict['AP_5_95']
                model_save_path = os.path.join(OUTPUT_DIR, "detr_ct_nodule_best_model.pth")
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"New best model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

        # Optionally, save model checkpoint every N epochs or at the end of each epoch
        if epoch % (config.save_every_n_epochs if hasattr(config, 'save_every_n_epochs') else 5) == 0:
            epoch_model_path = os.path.join(OUTPUT_DIR, f"detr_ct_nodule_epoch_{epoch}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            logger.info(f"Model checkpoint saved to {epoch_model_path}")


    logger.info("Training finished.")
    
    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, "detr_ct_nodule_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Save the image processor and model config for easy reloading later for inference
    # The image_processor is from the global scope in detr_data.py or re-initialized here.
    # For consistency, use the one that collate_fn uses.
    # from detr_data import IMAGE_PROCESSOR as data_module_image_processor
    # data_module_image_processor.save_pretrained(OUTPUT_DIR)
    # Or if you re-initialized it in train_detr.py:
    IMAGE_PROCESSOR.save_pretrained(OUTPUT_DIR)
    model.config.save_pretrained(OUTPUT_DIR)
    logger.info(f"Image processor and model configuration saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()