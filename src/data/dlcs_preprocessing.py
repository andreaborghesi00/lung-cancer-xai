import numpy as np
from typing import Tuple, List, Optional, Callable, Any, Union
import torch
from config.config_3d import get_config
import logging
import SimpleITK as stik
from skimage.segmentation import clear_border
from skimage import measure
from skimage.measure import label,regionprops
from scipy import ndimage as ndi
from scipy.ndimage import center_of_mass, binary_dilation
from monai.config import KeysCollection
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
from monai.data.box_utils import clip_boxes_to_image
from monai.apps.detection.transforms.box_ops import convert_box_to_mask
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Lambdad,
    Orientationd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    RandZoomd,
    RandFlipd,
    RandRotate90d,
    MapTransform,
)
from monai.transforms.utility.dictionary import ApplyTransformToPointsd
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
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
from monai.config import KeysCollection
from monai.utils.type_conversion import convert_data_type
from monai.data.box_utils import clip_boxes_to_image
from monai.apps.detection.transforms.box_ops import convert_box_to_mask



logger = logging.getLogger(__name__)
conf = get_config()
    
def load_itk_image(filename):
    '''
    this function load an image using simpleITK and return
    3D array, origin_numpy and spacing numpy
    input:  filename     = nifty CT image path eg."./CT_CTGRAV_TMP404180_4.nii.gz"
    output: numpyImage   = 3D CT numpy as slice X height x Width (z,y,x)
            numpyOrigin  = (z_origin,  y_origin,  x_origin)
            numpySpacing = (z_Spacing, y_Spacing, x_Spacing)
    '''
    itkimage = stik.ReadImage(filename)           # load the ct image
    numpyImage = stik.GetArrayFromImage(itkimage) # give CT to 3D numpy (z,y,x)
   
    #- When we load get the  itkimage.GetOrigin/GetSpacing it 
    # returns the array in order (x_origin, y_origin,z_origin)
    # as we loaded the CT image as 3D numpy (z,y,x) that is why
    # we also reversed the acquired spacing and origin
    numpyOrigin = np.array(itkimage.GetOrigin())[::-1]  # Convert to (z_origin, y_origin, x_origin)
    numpySpacing = np.array(itkimage.GetSpacing())[::-1]  # Convert to (z_spacing, y_spacing, x_spacing)


    # Example usage:
    # voxel_coords = world_to_voxel(world_coords)
    return numpyImage, numpyOrigin, numpySpacing

def normalize_tomography(volume: np.ndarray, lbound = -1200, ubound=300) -> np.ndarray:
    """
    Normalize the volume to have values between 0 and 1
    """
    volume = volume.astype(np.float32)
    volume = np.clip(volume, lbound, ubound)
    volume -= volume.min()
    volume /= volume.max()
    return volume

# utility function to convert world coordinates to voxel coordinates
def world_to_voxel(world_coords, origin, spacing):
    """
    Convert world coordinates to voxel coordinates
    world_coords: array-like, shape (N, 3) in (z, y, x) order
    origin: array-like, shape (3,) in (z, y, x) order
    spacing: array-like, shape (3,) in (z, y, x) order
    returns: array-like, shape (N, 3) in (z, y, x) order
    """
    return np.round((np.array(world_coords) - origin) / spacing).astype(int)


def keep_top_3(slc):
    new_slc = np.zeros_like(slc)
    rps = regionprops(slc)
    areas = [r.area for r in rps]
    idxs = np.argsort(areas)[::-1]
    for i in idxs[:3]:
        new_slc[tuple(rps[i].coords.T)] = i+1
    return new_slc


def delete_table(slc):
    new_slc = slc.copy()
    labels = label(slc, background=0)
    idxs = np.unique(labels)[1:]
    COM_ys = np.array([center_of_mass(labels==i)[0] for i in idxs])
    for idx, COM_y in zip(idxs, COM_ys):
        if (COM_y < 0.3*slc.shape[0]):  # se il centro di massa è più grandel del 30%, calcellalo. quindi toglie tutta la parte sotto
            new_slc[labels==idx] = 0
        elif (COM_y > 0.6*slc.shape[0]):  # allo stesso tempo, se il centro di massa è maggiore del 60%, cancellalo! quindi toglie qualsiasi cosa sopra. lo faccio perchè a volte può esserci qualcosa sopra, ma in teoria non nel mio caso.
            new_slc[labels==idx] = 0
    return new_slc


def compute_mask(volume: np.ndarray, normalized: bool = False, dilation_iters: int = 3) -> np.ndarray:
    if dilation_iters < 0: raise ValueError("dilation_iters must be >= 0")
    
    mask = volume < 0.23 if normalized else volume < -300 # TODO: double check lungs HU values
    mask = np.vectorize(clear_border, signature="(m,n)->(m,n)")(mask)
    mask = np.vectorize(measure.label, signature="(m,n)->(m,n)")(mask)
    mask = np.vectorize(keep_top_3, signature="(m,n)->(m,n)")(mask)
    mask = mask > 0
    mask = np.vectorize(ndi.binary_fill_holes, signature="(m,n)->(m,n)")(mask)
    mask = np.vectorize(delete_table, signature='(n,m)->(n,m)')(mask)
    
    for _ in range(dilation_iters): mask = np.vectorize(binary_dilation, signature='(n,m)->(n,m)')(mask)
    
    return mask


def get_tomography_paths(dataset_dir: str) -> List[str]:
    raise NotImplementedError("Not implemented yet")


def get_train_transforms(
    patch_size: Tuple[int, int, int],
    batch_size: int,
    image_key: str,
    box_key: str,
    label_key: str,
    label_mask_key: str,
    point_key: str,
    box_mask_key: str,
    affine_lps_to_ras: bool = False,
    gt_box_mode: str = "cccwhd"
    ):
    train_transform = Compose([
    LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict", reader="NumpyReader"),
    Lambdad(keys=[image_key], func=lambda x: np.transpose(x, (2, 1, 0))),  # D, H, W -> WHD (x, y, z)
    EnsureChannelFirstd(keys=[image_key], channel_dim='no_channel'),
    EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
    EnsureTyped(keys=[label_key], dtype=torch.long),
    StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
    ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
    ConvertBoxToPointsd(keys=[box_key], point_key=point_key),
    # GenerateBoxMask(
    #     keys=box_key,
    #     image_key=image_key,
    #     box_key=box_key,
    #     mask_image_key=label_mask_key,
    #     spatial_size=patch_size,
    # ),
    GenerateExtendedBoxMask( # this allows for the box to not always be around the center of the cropped image
        keys=box_key,
        image_key=image_key,
        spatial_size=patch_size,
        whole_box=True,
        mask_image_key=label_mask_key,
    ),
    RandCropByPosNegLabeld(
        keys=[image_key],
        label_key=label_mask_key,
        spatial_size=patch_size,
        num_samples=batch_size,
        pos=1,
        neg=0,
    ),
    ApplyTransformToPointsd(keys=[point_key],
                            refer_keys=image_key,
                            affine_lps_to_ras=affine_lps_to_ras,
                            ),
    ConvertPointsToBoxesd(keys=[point_key], box_key=box_key),      
    ClipBoxToImaged(
        box_keys=box_key,
        label_keys=[label_key],
        box_ref_image_keys=image_key,
        remove_empty=True,
    ),
    
    BoxToMaskd(
    box_keys=[box_key],
    label_keys=[label_key],
    box_mask_keys=[box_mask_key],
    box_ref_image_keys=[image_key],
    min_fg_label=0,
    ellipse_mask=True,
    ),
    RandRotated(
        keys=[image_key, box_mask_key],
        mode=["nearest", "nearest"],
        prob=0.2,
        range_x=np.pi / 6,
        range_y=np.pi / 6,
        range_z=np.pi / 6,
        keep_size=True,
        padding_mode="border",
    ),
    RandZoomd(
        keys=[image_key, box_mask_key],
        prob=0.2,
        min_zoom=0.7,
        max_zoom=1.4,
        padding_mode="constant",
        keep_size=True,
    ),
    RandFlipd(
        keys=[image_key, box_mask_key],
        prob=0.5,
        spatial_axis=0,
    ),
    RandFlipd(
        keys=[image_key, box_mask_key],
        prob=0.5,
        spatial_axis=1,
    ),
    RandFlipd(
        keys=[image_key, box_mask_key],
        prob=0.5,
        spatial_axis=2,
    ),
    RandRotate90d(
        keys=[image_key, box_mask_key],
        prob=0.75,
        max_k=3,
        spatial_axes=(0, 1),
    ),
    MaskToBoxd(
        box_keys=[box_key],
        label_keys=[label_key],
        box_mask_keys=[box_mask_key],
        min_fg_label=0,
    ),
    RandGaussianNoised(keys=[image_key], prob=0.1, mean=0, std=0.1),
    RandGaussianSmoothd(
        keys=[image_key],
        prob=0.1,
        sigma_x=(0.5, 1.0),
        sigma_y=(0.5, 1.0),
        sigma_z=(0.5, 1.0),
    ),
    RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.25),
    RandShiftIntensityd(keys=[image_key], prob=0.15, offsets=0.1),
    RandAdjustContrastd(keys=[image_key], prob=0.3, gamma=(0.7, 1.5)),
    
    
    DeleteItemsd(keys=[label_mask_key, point_key, "image_meta_dict", "image_meta_dict_meta_dict"]),
    EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
    ])
    
    return train_transform

def get_val_transforms(
    image_key: str,
    box_key: str,
    label_key: str,
    gt_box_mode: str = "cccwhd",
    ):
    
    val_transform = Compose([
        LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict", reader="NumpyReader"),
        Lambdad(keys=[image_key], func=lambda x: np.transpose(x, (2, 1, 0))),  # D, H, W -> WHD (x, y, z)
        EnsureChannelFirstd(keys=[image_key], channel_dim='no_channel'),
        EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
        EnsureTyped(keys=[label_key], dtype=torch.long),
        StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
        ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
        EnsureTyped(keys=[image_key, box_key], dtype=torch.float16),
    ])
    return val_transform

class GenerateBoxMask(MapTransform):
    """
    Deprecated.
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
        
    def generate_mask_img(self, boxes, image_size):
        mask_img = convert_box_to_mask(
            boxes, np.ones(boxes.shape[0]), image_size, bg_label=0, ellipse_mask=True
        )
        mask_img = np.amax(mask_img, axis=0, keepdims=True)[0:1, ...]
        return mask_img
    
    def __call__(self, data):
        image = data[self.image_key]
        boxes = data[self.box_key]
        data[self.mask_image_key] = self.generate_mask_img(boxes=boxes, image_size=image.shape[1:])
        
        return data

class GenerateExtendedBoxMask(MapTransform):
    """
    Yoinked from MONAI detection tutorial.
    Generate box mask based on the input boxes.
    By having a larger spatial size the cropper doesn't always put the box around the center of the image.
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
        return mask_img

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[self.image_key]
            boxes = d[key]
            data[self.mask_image_key] = self.generate_mask_img(boxes, image.shape[1:], whole_box=self.whole_box)
        return data


