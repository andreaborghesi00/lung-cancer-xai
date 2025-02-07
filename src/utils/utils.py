import numpy as np
import logging

def roi_overlay(image, roi, color=(255, 0, 0), lw=2):
    x, y, w, h = roi
    image = image.copy()
    image[y:y+h, x:x+lw] = color
    image[y:y+h, x+w:x+w+lw] = color
    image[y:y+lw, x:x+w] = color
    image[y+h:y+h+lw, x:x+w] = color
    return image


def greyscale_to_rgb(image):
    # return np.stack([image, image, image], axis=2)
    return np.repeat(image[..., np.newaxis], 3, axis=-1)

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )