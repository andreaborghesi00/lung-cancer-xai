import logging
from tqdm import tqdm
from pathlib import Path
import gc
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image

import utils.utils as utils
import models.monai_retinanet as rn
from config.config_3d import get_config
from data.dlcs_dataset import DLCSDataset
from data.dlcs_preprocessing import get_val_transforms_nifti 

def main():
    config = get_config()
    config.validate()
    logger = logging.getLogger(__name__)
    utils.setup_logging(level=logging.INFO)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cudnn.benchmark = True 
    torch.set_num_threads(4)
    
    checkpoint_path = Path("checkpoints/RetinaNet/dict_150_epochs_not_augmented_warmup_last.pt")

    val_transform = get_val_transforms_nifti(
    image_key=config.image_key,
    box_key=config.box_key,
    label_key=config.label_key,
    gt_box_mode=config.gt_box_mode,
    )

    annotations = pd.read_csv(config.annotations_path)

    pids = annotations["patient-id"].unique().tolist()
    benign_pids = annotations[annotations['Malignant_lbl'] == 1]["patient-id"].unique().tolist()
    malignant_pids = annotations[annotations['Malignant_lbl'] == 0]["patient-id"].unique().tolist()
    y = [1 if pid in benign_pids else 0 for pid in pids] # 1 if the patient has at least a benign nodule, 0 otherwise. Used to stratify the split
    logger.info(f"Unique patients: {len(pids)}\nOf which {len(benign_pids)} are benign and {len(malignant_pids)} are malignant")

    # split into train and val
    train_pids, val_pids = train_test_split(pids, test_size=0.2, stratify=y, random_state=config.split_seed)
    
    logger.info(f"Train patients: {len(train_pids)} [{len([i for i in train_pids if i in malignant_pids])} | {len([i for i in train_pids if i in benign_pids])}] | Val patients: {len(val_pids)} [{len([i for i in val_pids if i in malignant_pids])} | {len([i for i in val_pids if i in benign_pids])}]")
    
    train_annotations = annotations[annotations["patient-id"].isin(train_pids)]
    train_annotations.reset_index(drop=True, inplace=True)
    
    val_annotations = annotations[annotations["patient-id"].isin(val_pids)]
    val_annotations.reset_index(drop=True, inplace=True)
    
    val_ds = DLCSDataset(val_annotations, config.data_dir, transform=val_transform)
    
    # train_dl = train_ds.get_loader(shuffle=True, num_workers=config.dl_workers, batch_size=config.dl_batch_size) # careful with batch size
    # val_dl = val_ds.get_loader(shuffle=False, num_workers=config.dl_workers, batch_size=1)

    val_dl = val_ds.get_monai_loader(shuffle=False, num_workers=config.dl_workers, batch_size=1)
    
    
    detector = rn.create_retinanet_detector(
        device=device,
        pretrained=config.pretrained,
        # pretrained_path=None,
        n_input_channels=config.n_input_channels,
        base_anchor_shapes=config.base_anchor_shapes,
        conv1_t_stride=config.conv1_t_stride,
        num_classes=1,
        trainable_backbone_layers=config.trainable_backbone_layers
    )
    
    detector.network.load_state_dict(torch.load(checkpoint_path))
    
    target_layers = [list(detector.network.feature_extractor.body.children())[-1]]
    explainer = GradCAM(model=detector, target_layers=target_layers)

    detector.eval()
    val_outputs_all = []
    val_targets_all = []
    val_inputs_all = []
    val_pbar = tqdm(val_dl, total=len(val_dl))
    for val_data in val_pbar:
        # val_inputs = [val_data.pop("image").squeeze(0).to(device)] # we pop so that val data now contains only boxes and labels
        val_inputs = [val_data_i.pop("image").to(device) for val_data_i in val_data]
        val_inputs[0].requires_grad = True
        
        with torch.autocast("cuda"):
            val_outputs = detector(val_inputs, use_inferer=True) # only inference

        # save outputs for evaluation
        val_outputs_all += val_outputs 
        val_targets_all += val_data # here are the ground truths (without the image)
        val_inputs_all += val_inputs
        
        cam = explainer(val_inputs[0], targets=None)
        vis = show_cam_on_image(val_inputs[0].cpu().numpy(), cam[0].cpu().numpy(), use_rgb=True)
        plt.imshow(vis)
        plt.axis("off")
        plt.savefig(f"cam.png", bbox_inches="tight", pad_inches=0)

if __name__ == "__main__":
    main()