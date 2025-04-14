import multiprocessing
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    SaveImaged,
)
import torch
from tqdm import tqdm

data_dir = "/gwdata/users/aborghesi/DLCS/subset_1_to_6/"
output_dir = "/gwdata/users/aborghesi/DLCS/subset_1_to_6_resampled/"

transform = Compose([
    LoadImaged(keys=["image"],reader="NibabelReader", image_only=False, meta_key_postfix="meta_dict"),
    EnsureChannelFirstd(keys=["image"]),
    EnsureTyped(keys=["image"], dtype=torch.float16),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=['image'], pixdim=([0.703125, 0.703125, 1.25]), padding_mode="border"),
])

save_transform = Compose([
    SaveImaged(
        keys="image",
        meta_keys=["image_meta_dict"],
        output_dir=Path(output_dir),
        output_postfix="",
        resample=False,
        separate_folder=False,
    )
])

def process_file(path):
    """Worker function to process a single file"""
    print(f"Processing {path}")
    transformed_data = transform({"image": str(path)})
    save_transform(transformed_data)
    return True

def main(num_processes=None):
    # Get all .gz files
    files = [path for path in Path(data_dir).iterdir() 
             if path.is_file() and path.suffix == ".gz"]
    
    # Set default number of processes to CPU count - 1 (or 1 if only 1 CPU)
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # Create pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files)))
    
    print(f"Processed {len(results)} files")

if __name__ == "__main__":
    main(num_processes=6)