import logging
import random
from config.config import get_config
from data.rcnn_dataset import DynamicRCNNDataset
import utils.utils as utils
from models.faster_rcnn import FasterRCNNMobileNet, FasterRCNNResnet50
from explainers.grad_cam import CAMExplainer
from utils.visualization import Visualizer
from tqdm import tqdm
import torch
from pathlib import Path
from data.preprocessing import ROIPreprocessor
from sklearn.model_selection import train_test_split
import gc
from PIL import Image
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    
    preprocessor = ROIPreprocessor()
    unique_tomographies = preprocessor.load_tomography_ids()
    
    # split unique tomographies into train, validation and test
    logger.info("Splitting data into train, validation and test sets")
    train_ids, valtest_ids = train_test_split(unique_tomographies, test_size=(1-config.train_split_ratio), random_state=config.random_state)
    val_ids, test_ids = train_test_split(valtest_ids, test_size=(1-config.val_test_split_ratio), random_state=config.random_state)

    # load paths and labels
    logger.info("Loading paths and labels")
    X_train, y_train = preprocessor.load_paths_labels(train_ids)
    X_val, y_val = preprocessor.load_paths_labels(val_ids)
    X_test, y_test = preprocessor.load_paths_labels(test_ids)
    
    # labels to numpy
    y_train = [label.numpy() for label in y_train]
    y_val = [label.numpy() for label in y_val]
    y_test = [label.numpy() for label in y_test]
    
    # save images and labels for each set to disk according to yolo format
    images = [X_train, X_val, X_test]
    labels = [y_train, y_val, y_test]
    sets = ["train", "val", "test"]
    for image_set, set_name in zip(images, sets):
        image_path = Path(config.yolo_dataset_dir) / f"images" / set_name
        for i, image in tqdm(enumerate(image_set)):
            image_yolo = Image.open(image)
            image_numpy = np.array(image_yolo)
            image_numpy = preprocessor.normalize_background(image_numpy)
            image_yolo = Image.fromarray(image_numpy)
            image_yolo.save(image_path / f"{i}.png")
    
    logger.info("Saving labels to yolo format")
    logger.warning("Assuming that there is only one bounding box per image!")
    for label_set, set_name in zip(labels, sets):
        label_path = Path(config.yolo_dataset_dir) / f"labels" / set_name
        for i, label in tqdm(enumerate(label_set)):
            label_yolo = preprocessor.xyxy_to_yolo(label)
            label_yolo = preprocessor.normalize_bbox(label_yolo)
            with open(label_path / f"{i}.txt", "w") as f:
                # we're assuming there's only one bounding box per image
                f.write(f"0 {label_yolo[0]} {label_yolo[1]} {label_yolo[2]} {label_yolo[3]}\n") # 0 is the class, the rest are the bounding box coordinates
             
    
    
    
    