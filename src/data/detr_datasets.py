from config.config_2d import get_config
import os
# import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from transformers.image_utils import AnnotationFormat, ChannelDimension
import albumentations as A
# from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
# import random # For train/val split
import pandas as pd
import logging
import matplotlib.pyplot as plt

# --- Logger ---
logger = logging.getLogger(__name__)

# --- Configuration ---
config = get_config()
config.validate()
# Define your classes
# For simplicity, let's first detect "nodule" (class 0) and then "benign" (class 0) vs "malignant" (class 1)
# If you only want to detect "nodule" as one class:
# CLASS_NAMES = ["nodule"]
# If you want to distinguish benign/malignant:
# CLASS_NAMES = ["benign_nodule", "malignant_nodule"] # Or just "nodule" if not distinguishing
CLASS_NAMES = ["nodule"] # Or just "nodule" if not distinguishing
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

CSV_PATIENT_ID_COL = "pid"
CSV_NODULE_ID_COL = "nodule_id" # Not directly used by DETR target format
CSV_IS_BENIGN_COL = "is_benign"
CSV_XMIN_COL = "bbox_x1"
CSV_YMIN_COL = "bbox_y1"
CSV_XMAX_COL = "bbox_x2"
CSV_YMAX_COL = "bbox_y2"
CSV_IMAGE_PATH_COL = "path" 

# DETR model expects num_labels = number of object classes (it adds the "no object" class internally)
MODEL_CHECKPOINT = "facebook/detr-resnet-50"
IMAGE_PROCESSOR = DetrImageProcessor.from_pretrained(MODEL_CHECKPOINT)
# IMAGE_PROCESSOR.size = None
IMAGE_PROCESSOR.do_rescale = False
IMAGE_PROCESSOR.do_resize = False
IMAGE_PROCESSOR.do_normalize = False
IMAGE_PROCESSOR.format = AnnotationFormat.COCO_DETECTION
DO_PAD = True # Set to False if you want to pad to the max size of the batch
PAD_SIZE = {"height": 750, "width": 750} # (height, width) for padding
# Training params
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4 # Adjust based on your GPU memory
NUM_EPOCHS = 10 # Start with a few, increase later
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_annotations_from_csv(annotation_csv_path, class_to_id_map):
    """
    Loads annotations from a CSV file and prepares them in a COCO-like format,
    grouped by image path.
    Assumes the CSV contains a column with direct paths to the .npy image files.
    """
    try:
        df_annotations = pd.read_csv(config.annotation_path)
    except FileNotFoundError:
        logger.error(f"Annotation CSV file not found at {config.annotation_path}")
        return []
    except Exception as e:
        logger.error(f"reading CSV file {config.annotation_path}: {e}")
        return []

    # Check for required columns
    required_cols = [CSV_PATIENT_ID_COL, CSV_IS_BENIGN_COL, CSV_XMIN_COL, CSV_YMIN_COL, CSV_XMAX_COL, CSV_YMAX_COL, CSV_IMAGE_PATH_COL]
    for col in required_cols:
        if col not in df_annotations.columns:
            logger.error(f"Required column '{col}' not found in the annotation CSV.")
            logger.error(f"Available columns are: {df_annotations.columns.tolist()}")
            return []

    image_id_counter = 0
    image_path_to_id_map = {}
    prepared_data = {} # image_path -> { 'image_id': int, 'annotations': list_of_coco_annos }

    logger.info(f"Processing {len(df_annotations)} annotation entries from CSV...")
    for index, row in tqdm(df_annotations.iterrows(), total=df_annotations.shape[0]):
        try:
            # patient_id = str(row[CSV_PATIENT_ID_COL]) # For debugging or other metadata
            is_benign = bool(row[CSV_IS_BENIGN_COL])
            
            # Ensure bbox coordinates are numeric and handle potential errors
            xmin = int(float(row[CSV_XMIN_COL]))
            ymin = int(float(row[CSV_YMIN_COL]))
            xmax = int(float(row[CSV_XMAX_COL]))
            ymax = int(float(row[CSV_YMAX_COL]))

            image_path = str(row[CSV_IMAGE_PATH_COL]).strip() # Get the direct image path

        except ValueError as e:
            logger.warning(f"Skipping row {index} due to invalid numeric value for bbox: {e}. Row data: {row.to_dict()}")
            continue
        except KeyError as e:
            logger.warning(f"Skipping row {index} due to missing column: {e}. Row data: {row.to_dict()}")
            continue


        # --- Define your class_id based on is_benign ---
        if "benign_nodule" in class_to_id_map and "malignant_nodule" in class_to_id_map:
            class_name = "benign_nodule" if is_benign else "malignant_nodule"
        elif "nodule" in class_to_id_map:
            class_name = "nodule"
        else:
            # This should not happen if CLASS_NAMES and class_to_id_map are set up correctly
            logger.warning(f"Class definition mismatch for row {index}. Skipping.")
            continue
        category_id = class_to_id_map[class_name]
        # --- ---

        if not os.path.exists(image_path):
            logger.warning(f"Image file not found at path '{image_path}' specified in CSV row {index}. Skipping annotation.")
            continue

        # Assign a unique integer ID to this image if not seen before
        # The image_path itself is the unique key for an image
        if image_path not in image_path_to_id_map:
            image_path_to_id_map[image_path] = image_id_counter
            current_image_id = image_id_counter
            image_id_counter += 1
        else:
            current_image_id = image_path_to_id_map[image_path]

        # Bounding box in COCO format: [xmin, ymin, width, height]
        width = xmax - xmin
        height = ymax - ymin
        
        if width <= 0 or height <= 0:
            # print(f"Warning: Skipping annotation with non-positive width/height for image {image_path} (row {index}). bbox: {[xmin, ymin, width, height]}")
            continue

        bbox_coco = [xmin, ymin, width, height]
        area = width * height

        annotation_coco_format = {
            "image_id": current_image_id,
            "category_id": category_id,
            "bbox": bbox_coco,
            "area": area,
            "iscrowd": 0,
        }
        
        if image_path not in prepared_data:
            prepared_data[image_path] = {"image_id": current_image_id, "annotations": []}
        
        prepared_data[image_path]["annotations"].append(annotation_coco_format)

    # Convert prepared_data dict to a list of (image_path, target_dict) for the Dataset
    dataset_items = []
    for img_path, data in prepared_data.items():
        target_for_processor = {"image_id": data["image_id"], "annotations": data["annotations"]}
        dataset_items.append((img_path, target_for_processor))
        
    logger.info(f"Prepared {len(dataset_items)} unique images with annotations from CSV.")
    if not dataset_items:
        logger.warning("No valid dataset items were prepared. Check your CSV data, paths, and column names.")
    return dataset_items

class CTNoduleDataset(Dataset):
    def __init__(self, dataset_items, transforms=None): # Removed image_processor from here
        self.dataset_items = dataset_items
        self.transforms = transforms # Albumentations transforms
        self.max_hu = 400
        self.min_hu = -1000
        
    def clip_normalize_hu(self, image_array):
        image_array = np.clip(image_array, self.min_hu, self.max_hu)
        return (image_array - self.min_hu) / (self.max_hu - self.min_hu)
    
    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        image_path, target_coco_format = self.dataset_items[idx] # target_coco_format is already good
        
        image_array = np.load(image_path)
        if image_array.ndim == 3 and image_array.shape[-1] == 1:
            image_array = image_array.squeeze(-1)
        
        # --- Proper CT Windowing (CRUCIAL) ---
        image_array = self.clip_normalize_hu(image_array) # Normalize to 0-1 given a cer
        # image_uint8 = (image_array * 255).astype(np.uint8)
        # --- End Windowing ---

        image_rgb = np.stack([image_array]*3, axis=-1) # Convert to (H, W, 3)

        # save the image for debugging
        # bbox = target_coco_format['annotations'][0]['bbox']
        # plt.imshow(image_rgb)
        # plt.axis('off')
        # plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor='red', facecolor='none', lw=2))
        # plt.savefig(f"detr_output/pre_albumentations_{idx}.png")
        # plt.close()
        
        # Prepare for Albumentations
        bboxes_for_alb = []
        class_labels_for_alb = []
        original_image_height, original_image_width = image_rgb.shape[:2]

        for ann in target_coco_format['annotations']:
            coco_bbox = ann['bbox'] # [xmin, ymin, width, height]
            x_min, y_min, w, h = coco_bbox
            # Convert to [xmin, ymin, xmax, ymax] for albumentations
            x_max = x_min + w
            y_max = y_min + h
            
            # Clip boxes
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(original_image_width, x_max)
            y_max = min(original_image_height, y_max)

            if x_min >= x_max or y_min >= y_max: continue

            bboxes_for_alb.append([x_min, y_min, x_max, y_max])
            class_labels_for_alb.append(ann['category_id'])

        # Apply Albumentations transforms
        if self.transforms:
            transformed = self.transforms(image=image_rgb, 
                                          bboxes=bboxes_for_alb, 
                                          class_labels=class_labels_for_alb)
            image_transformed_np = transformed['image'] # This is numpy array (H,W,C) or (C,H,W)
            transformed_bboxes_abs = transformed['bboxes'] # [xmin, ymin, xmax, ymax]
            transformed_class_labels = transformed['class_labels']

            # Update target annotations with transformed bboxes (COCO format)
            new_annotations_coco = []
            for i, t_bbox_abs in enumerate(transformed_bboxes_abs):
                new_ann = {
                    "image_id": target_coco_format["image_id"],
                    "category_id": transformed_class_labels[i],
                    "bbox": [t_bbox_abs[0], t_bbox_abs[1], t_bbox_abs[2] - t_bbox_abs[0], t_bbox_abs[3] - t_bbox_abs[1]],
                    "area": (t_bbox_abs[2] - t_bbox_abs[0]) * (t_bbox_abs[3] - t_bbox_abs[1]),
                    "iscrowd": 0,
                }
                new_annotations_coco.append(new_ann)
            target_coco_format['annotations'] = new_annotations_coco
            
            final_image_to_process = image_transformed_np # let's pass a numpy rather than PIL
        else:
            final_image_to_process = image_rgb # also here, we pass a numpy rather than PIL
        
        # bbox_new  target_coco_format['annotations'][0]['bbox']
        # plt.imshow(final_image_to_process)
        # plt.axis('off')
        # plt.gca().add_patch(plt.Rectangle((bbox_new[0], bbox_new[1]), bbox_new[2], bbox_new[3], edgecolor='red', facecolor='none', lw=2))
        # plt.savefig(f"detr_output/pre_detr_image_processor_{idx}.png")
        # plt.close()
        
        # DetrImageProcessor expects image_id in the target, and 'annotations' as a list of dicts.
        # It also needs 'orig_size' and 'size' for postprocessing.
        # These will be added by the DetrImageProcessor itself.
        
        return final_image_to_process, target_coco_format


# Albumentations transforms
# Note: DetrImageProcessor will handle resizing to DETR's required input size and normalization.
# So, Albumentations resize here is optional or for initial consistent sizing if needed.
# BboxParams format 'coco' is [x_min, y_min, width, height].
# 'pascal_voc' is [x_min, y_min, x_max, y_max]. We use pascal_voc for albumentations internal,
# then convert back to COCO for DETR processor. why tho 
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.CLAHE(p=1.0)

], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

val_transforms = A.Compose([
    A.CLAHE(p=1.0),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))



def collate_fn(batch):
    # batch is a list of (numpy matrix, target_coco_format) tuples
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    batch_processed = IMAGE_PROCESSOR(images=pixel_values,
                                      annotations=labels, 
                                      return_tensors="pt", 
                                      do_pad=DO_PAD, 
                                      pad_size=PAD_SIZE,
                                    #   input_data_format=ChannelDimension.LAST,
                                    #   data_format=ChannelDimension.LAST
                                      )
    return batch_processed


if __name__ == "__main__":
    CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

    print(load_and_prepare_annotations_from_csv(config.annotation_path, CLASS_TO_ID))