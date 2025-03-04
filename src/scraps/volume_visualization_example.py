 
import pydicom
import numpy as np
import cv2 as cv
from pathlib import Path
import os
import SimpleITK as stik
import matplotlib.pyplot as plt

"""
If you are running this through ssh and want to display the plots on your local machine,
ensure that you have X11 forwarding enabled in your ssh config file, or use the -X (or -Y) flag when sshing.
e.g. ssh -X user@host
and run this on that terminal, avoid running this on notebooks or VScode/IDE terminals
"""

plt.ion() # interactive mode on

class ScrollPlot:
    def __init__(self,
                 volume: np.ndarray
                 ) -> None:
        self.volume = volume
        self.slices = volume.shape[0] # depth of the volume (number of slices)
        self.index = self.slices // 2  # Start in the middle
        
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.volume[self.index], cmap="gray")
        self.update() # draw the first slice
        
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll) # connect the scroll event, call on_scroll when scrolling
        plt.show(block=True)

    def update(self):
        self.img.set_data(self.volume[self.index]) # update the image
        self.ax.set_title(f"Slice {self.index + 1}/{self.slices}") # update the title
        self.ax.figure.canvas.draw() # redraw the canvas

    def on_scroll(self, event):
        if event.step > 0:  # scroll up
            self.index = min(self.index + 1, self.slices - 1) # next slice, if last stay in the last
        else:  # Scroll down
            self.index = max(self.index - 1, 0) # previous slice, if first stay in the first
        self.update() # update the plot

 
def load_dicom(path: str, normalize: bool = True):
    """
    Load a dicom file and return the pixel array as a numpy array
    
    Raises FileNotFoundError if the file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    dicom = pydicom.dcmread(path)
    pixel_array = dicom.pixel_array.astype(np.float32)
    
    #rescale
    if "RescaleSlope" in dicom and "RescaleIntercept" in dicom:
        print("rescaling dicom")
        pixel_array = pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept

    #normalize
    if normalize:
        pixel_array -= pixel_array.min()
        pixel_array /= pixel_array.max()
    return pixel_array

 
def load_dicom_series(dir: str):
    reader = stik.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dir)
    dicom_files = reader.GetGDCMSeriesFileNames(dir, series_ids[0]) # only load the first series
    
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    return stik.GetArrayFromImage(image)

 
def to_uint8(image: np.ndarray):
    """
    Convert a numpy array to uint8
    """
    return (image * 255).astype(np.uint8)

 
# print cwd
print(os.getcwd())

 
dataset_dir = Path("../datasets")
batch = Path("set1_batch1_A/set1_batch1/sub_batch_A")
patients_dir = dataset_dir / batch

 
# take the first tomography of the first patient
patient = patients_dir / "127399"
tomography_dir = patient / "T0"

 
print(tomography_dir.iterdir().__next__())

 
# load the first dicom file
dicom_file_path = tomography_dir.iterdir().__next__()
dicom = load_dicom_series(dicom_file_path)
print(dicom.shape)

 
ScrollPlot(dicom)

 



