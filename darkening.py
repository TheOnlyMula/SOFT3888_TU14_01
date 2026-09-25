from numbers import Real
import numpy as np
import torch
import kornia
import argparse
from pathlib import Path
import cv2

from PIL import Image
from numpy import asarray

STRENGTH_LEVELS = {"mild": 0, "strong": 0}

def save_rgb(path: Path, img: np.ndarray):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Could not encode {path}")
    encoded.tofile(str(path))  # Supports non-ASCII Windows paths
   

def darken(img):
    img_tensor = img.squeeze(0)
    img_linear = kornia.color.rgb_to_linear_rgb(img_tensor)

    naive_darkening_constant = 0.3
    naive_darken = naive_darkening_constant * img_linear
    
    img_final = kornia.color.linear_rgb_to_rgb(naive_darken)

    arr_final = img_final.detach().cpu().permute(1, 2, 0).numpy()
    arr_final = (arr_final * 255).clip(0, 255).astype(np.uint8)
    return arr_final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True, help="An 8-bit, three-channel RGB image file")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; existing folders are not overwritten")
    #args = parser.parse_args()

    img = Image.open('D:/DSEC_CAPSTONE/train/interlaken_00_c/images/left/distorted/000003.png') # Load image
    img_tensor = torch.from_numpy(asarray(img)) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)

    result = darken(img_tensor)
    save_rgb('D:/_res.png', result)




if __name__ == "__main__":
    main()