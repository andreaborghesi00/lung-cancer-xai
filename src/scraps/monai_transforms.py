import csv
import math
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

from skimage.segmentation import clear_border
from skimage import measure
from skimage.measure import label,regionprops
from scipy import ndimage as ndi
from scipy.ndimage import measurements, center_of_mass, binary_dilation, zoom
import torch

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, ScaleIntensityRanged, RandRotate90d, RandFlipd,
    RandCropByPosNegLabeld, ToTensord, CastToTyped, SpatialPadd, Lambdad, apply_transform,
    MapTransform
)

from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    ConvertBoxModed,
    StandardizeEmptyBoxd,
    
)
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.transforms.utility.dictionary import ApplyTransformToPointsd

from monai.config import KeysCollection
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
from monai.data.box_utils import clip_boxes_to_image
from monai.apps.detection.transforms.box_ops import convert_box_to_mask
from monai.utils import set_determinism

annotations = pd.read_csv("../DLCS/DLCSD24_Annotations_voxel.csv")
dataset_dir = Path("../DLCS/subset_01_processed")

annotations.sort_values(by="patient-id", inplace=True)
annotations.reset_index(drop=True, inplace=True)

sample_id = 4
pid = annotations["patient-id"].unique().tolist()[sample_id]
filename = dataset_dir / f"{str(pid)}.npy"
patient_df = annotations[annotations["patient-id"] == pid]

boxes = torch.tensor(patient_df[['coordX', 'coordY', 'coordZ', 'w', 'h', 'd']].values)
labels = torch.tensor(patient_df["Malignant_lbl"].values, dtype=torch.long) 

data = {
    "image": filename,
    "box":boxes,
    "label":labels,
    }

image_key = "image"
box_key = "box"
label_key = "label"
point_key = "points"
label_mask_key = "label_mask"

class GenerateExtendedBoxMask(MapTransform):
    """
    Generate box mask based on the input boxes.
    """

    def __init__(
        self,
        keys: KeysCollection,
        image_key: str,
        spatial_size: tuple[int, int, int],
        whole_box: bool,
        mask_image_key: str = "mask_image",
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            image_key: key for the image data in the dictionary.
            spatial_size: size of the spatial dimensions of the mask.
            whole_box: whether to use the whole box for generating the mask.
            mask_image_key: key to store the generated box mask.
        """
        super().__init__(keys)
        self.image_key = image_key
        self.spatial_size = spatial_size
        self.whole_box = whole_box
        self.mask_image_key = mask_image_key

    def generate_fg_center_boxes_np(self, boxes, image_size, whole_box=True):
        # We don't require crop center to be within the boxes.
        # As along as the cropped patch contains a box, it is considered as a foreground patch.
        # Positions within extended_boxes are crop centers for foreground patches
        spatial_dims = len(image_size)
        boxes_np, *_ = convert_data_type(boxes, np.ndarray)

        extended_boxes = np.zeros_like(boxes_np, dtype=int)
        boxes_start = np.ceil(boxes_np[:, :spatial_dims]).astype(int) 
        boxes_stop = np.floor(boxes_np[:, spatial_dims:]).astype(int)
        for axis in range(spatial_dims):
            if not whole_box:
                extended_boxes[:, axis] = boxes_start[:, axis] - self.spatial_size[axis] // 2 + 1
                extended_boxes[:, axis + spatial_dims] = boxes_stop[:, axis] + self.spatial_size[axis] // 2 - 1
            else:
                # extended box start
                extended_boxes[:, axis] = boxes_stop[:, axis] - self.spatial_size[axis] // 2 - 1
                extended_boxes[:, axis] = np.minimum(extended_boxes[:, axis], boxes_start[:, axis])
                # extended box stop
                extended_boxes[:, axis + spatial_dims] = extended_boxes[:, axis] + self.spatial_size[axis] // 2
                extended_boxes[:, axis + spatial_dims] = np.maximum(
                    extended_boxes[:, axis + spatial_dims], boxes_stop[:, axis]
                )
        extended_boxes, _ = clip_boxes_to_image(extended_boxes, image_size, remove_empty=True)  # type: ignore
        return extended_boxes

    def generate_mask_img(self, boxes, image_size, whole_box=True):
        extended_boxes_np = self.generate_fg_center_boxes_np(boxes, image_size, whole_box)
        mask_img = convert_box_to_mask(
            extended_boxes_np, np.ones(extended_boxes_np.shape[0]), image_size, bg_label=0, ellipse_mask=False
        )
        mask_img = np.amax(mask_img, axis=0, keepdims=True)[0:1, ...]
        # print(f"mask shape: {mask_img.shape}")
        return mask_img

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[self.image_key]
            boxes = d[key]
            data[self.mask_image_key] = self.generate_mask_img(boxes, image.shape[1:], whole_box=self.whole_box)
        print(f"mask shape: {data[self.mask_image_key].shape}")
        print(f"mask type: {type(data[self.mask_image_key])}")
        print(f"mask dtype: {data[self.mask_image_key].dtype}")
        return data

class GenerateBoxMask(MapTransform):
    """
    Generate box mask based on the input boxes.
    """

    def __init__(
        self,
        keys: KeysCollection,
        image_key: str,
        box_key: str,
        spatial_size: tuple[int, int, int],
        mask_image_key: str = "mask_image",
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            image_key: key for the image data in the dictionary.
            spatial_size: size of the spatial dimensions of the mask.
            whole_box: whether to use the whole box for generating the mask.
            mask_image_key: key to store the generated box mask.
        """
        super().__init__(keys)
        self.image_key = image_key
        self.spatial_size = spatial_size
        self.mask_image_key = mask_image_key
        self.box_key = box_key
        
    def generate_fg_center_boxes_np(self, boxes, image_size, whole_box=True):
        # We don't require crop center to be within the boxes.
        # As along as the cropped patch contains a box, it is considered as a foreground patch.
        # Positions within extended_boxes are crop centers for foreground patches
        spatial_dims = len(image_size)
        print(boxes)
        boxes_np, *_ = convert_data_type(boxes, np.ndarray)

        extended_boxes = np.zeros_like(boxes_np, dtype=int)
        boxes_start = np.ceil(boxes_np[:, :spatial_dims]).astype(int) 
        boxes_stop = np.floor(boxes_np[:, spatial_dims:]).astype(int)
        print(boxes_start)
        print(boxes_stop)
        for axis in range(spatial_dims):
            if not whole_box:
                extended_boxes[:, axis] = boxes_start[:, axis] - self.spatial_size[axis] // 2 + 1
                extended_boxes[:, axis + spatial_dims] = boxes_stop[:, axis] + self.spatial_size[axis] // 2 - 1
            else:
                # extended box start
                extended_boxes[:, axis] = boxes_stop[:, axis] - self.spatial_size[axis] // 2 - 1
                extended_boxes[:, axis] = np.minimum(extended_boxes[:, axis], boxes_start[:, axis])
                # extended box stop
                extended_boxes[:, axis + spatial_dims] = extended_boxes[:, axis] + self.spatial_size[axis] // 2
                extended_boxes[:, axis + spatial_dims] = np.maximum(
                    extended_boxes[:, axis + spatial_dims], boxes_stop[:, axis]
                )
        extended_boxes, _ = clip_boxes_to_image(extended_boxes, image_size, remove_empty=True)  # type: ignore
        return extended_boxes
    
    def create_mask_from_boxes(self, image_shape, boxes):
        """
        Create a mask from the given boxes

        Args:
            image_shape: shape of the image (H, W, D)
            boxes: list boxes in the format (x1, y1, z1, x2, y2, z2)
        """
        mask = np.zeros(image_shape, dtype=np.int16)
        boxes_np, *_ = convert_to_dst_type(src=boxes, dst=mask)
        for box in boxes_np:
            x1, y1, z1, x2, y2, z2 = box
            print(f"Original box: x1={x1}, y1={y1}, z1={z1}, x2={x2}, y2={y2}, z2={z2}")
            
            x1 = max(0, x1)  
            y1 = max(0, y1)
            z1 = max(0, z1)
            x2 = min(image_shape[1], x2)  # W 
            y2 = min(image_shape[0], y2)  # H
            z2 = min(image_shape[2], z2)  # D
            print(f"Clipped box: x1={x1}, y1={y1}, z1={z1}, x2={x2}, y2={y2}, z2={z2}")

            mask[y1:y2, x1:x2, z1:z2] = 1 # 
        return mask[np.newaxis, ...]  # add channel dimension
    
    def generate_mask_img(self, boxes, image_size):
        # extended_boxes_np = self.generate_fg_center_boxes_np(boxes, image_size, whole_box)
        mask_img = convert_box_to_mask(
            boxes, np.ones(boxes.shape[0]), image_size, bg_label=0, ellipse_mask=True
        )
        print(f"mask shape: {mask_img.shape}")
        mask_img = np.amax(mask_img, axis=0, keepdims=True)[0:1, ...]
        return mask_img
    
    def __call__(self, data):
        image = data[self.image_key]
        boxes = data[self.box_key]
        # extended_boxes = self.generate_fg_center_boxes_np(boxes, image.shape[1:], whole_box=False)
        data[self.mask_image_key] = self.create_mask_from_boxes(image_shape=image.shape[1:], boxes=boxes)
        # data[self.mask_image_key] = self.generate_mask_img(boxes=boxes, image_size=image.shape[1:])
        print(f"mask shape: {data[self.mask_image_key].shape}")
        print(f"mask type: {type(data[self.mask_image_key])}")
        print(f"mask dtype: {data[self.mask_image_key].dtype}")
        
        return data


gt_box_mode = "cccwhd"
affine_lps_to_ras = False
patch_size = (192, 192, 80)
batch_size = 4

transform = Compose([
    LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict", reader="NumpyReader"),
    Lambdad(keys=[image_key], func=lambda x: np.transpose(x, (2,1,0))), # D, H, W -> H, W, D
    EnsureChannelFirstd(keys=[image_key], channel_dim='no_channel'),
    EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
    EnsureTyped(keys=[label_key], dtype=torch.long),
    StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
    Orientationd(keys=[image_key], axcodes="RAS"),
    ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
    ConvertBoxToPointsd(keys=[box_key], point_key=point_key),
    AffineBoxToImageCoordinated(
        box_keys=[box_key],
        box_ref_image_keys=image_key,

        image_meta_key_postfix="meta_dict",
        affine_lps_to_ras=affine_lps_to_ras,
    ),
    # RandRotate90d(
    #     keys=[image_key],
    #     prob=1,
    #     max_k=3,
    #     spatial_axes=(0, 1),
    # ),
    RandFlipd(
        keys=[image_key],
        prob=1,
        spatial_axis=1,
    ),
    # RandFlipd(
    #     keys=[image_key],
    #     prob=0.5,
    #     spatial_axis=1,
    # ),
    
    # GenerateBoxMask(
    #     keys=box_key,
    #     image_key=image_key,
    #     spatial_size=patch_size,
    #     box_key=box_key,
    #     mask_image_key=label_mask_key,
    #     # whole_box=True
    # ),
    # RandCropByPosNegLabeld(
    #     keys=[image_key],
    #     label_key=label_mask_key,
    #     spatial_size=patch_size,
    #     num_samples=batch_size,
    #     pos=1,
    #     neg=0,
    #     image_key=image_key
    # ),
    ApplyTransformToPointsd(keys=[point_key],
                            refer_keys=image_key, 
                            affine_lps_to_ras=affine_lps_to_ras,
                            ),
    ConvertPointsToBoxesd(keys=[point_key]),
    # ClipBoxToImaged(
    #     box_keys=box_key,
    #     label_keys=[label_key],
    #     box_ref_image_keys=image_key,
    #     remove_empty=True,
    # ),
    # EnsureTyped(keys=[image_key], dtype=torch.float32),
    # EnsureTyped(keys=[label_key], dtype=torch.long),

])

data_transformed = apply_transform(transform, data)
# data_transformed['box'], data_transformed.keys()
# [d["box"].int() for d in data_transformed]

# batch_id = 3
# sub_volume = data_transformed[batch_id][image_key].squeeze(0).numpy()
# boxes = data_transformed[batch_id][box_key].int().numpy()

# print(f"Sub-volume shape: {sub_volume.shape}")
# print(f"Boxes shape: {boxes.shape}")

# plt.figure(figsize=(16, 8*len(boxes)//2 + 6))

# for i, box in enumerate(boxes):
#     xmin, ymin, zmin, xmax, ymax, zmax = box
#     center_depth = (zmin + zmax) // 2
    
#     # Create subplot
#     plt.subplot(len(boxes), 1, i+1)
    
#     # Display slice at the center depth of the current box
#     plt.imshow(sub_volume[:, :, center_depth], cmap="gray")
#     plt.title(f"Box {i+1} at depth {center_depth}")
    
#     # Calculate box dimensions
#     w = int(xmax - xmin)
#     h = int(ymax - ymin)
    
#     # Draw rectangle for the current box
#     plt.gca().add_patch(plt.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='r', facecolor='none'))
    
#     # Add text annotation
#     plt.text(xmin, ymin-10, f"Box {i+1}: ({xmin}, {ymin})", color='r', fontsize=10, 
#              bbox=dict(facecolor='white', alpha=0.7))
    
#     plt.axis("off")

# plt.tight_layout()
# plt.show()

volume = data_transformed[image_key].squeeze(0).numpy()
boxes = data_transformed[box_key].int().numpy()

plt.figure(figsize=(16, 8*len(boxes)//2 + 6))

for i, box in enumerate(boxes):
    xmin, ymin, zmin, xmax, ymax, zmax = box
    center_depth = (zmin + zmax) // 2
    
    # Create subplot
    plt.subplot(len(boxes), 1, i+1)
    
    # Display slice at the center depth of the current box
    plt.imshow(volume[:, :, center_depth], cmap="gray")
    plt.title(f"Box {i+1} at depth {center_depth}")
    
    # Calculate box dimensions
    w = int(xmax - xmin)
    h = int(ymax - ymin)
    
    # Draw rectangle for the current box
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='r', facecolor='none'))
    
    # Add text annotation
    plt.text(xmin, ymin-10, f"Box {i+1}: ({xmin}, {ymin})", color='r', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis("off")

plt.tight_layout()
plt.show()




