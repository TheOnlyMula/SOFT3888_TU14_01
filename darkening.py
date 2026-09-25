from numbers import Real
import numpy as np
import torch
import kornia
import argparse
from pathlib import Path
import cv2

from PIL import Image
from numpy import asarray

STRENGTH_RANGES = {"minimal": (0.8, 0.95), "mild": (0.6, 0.8), "medium": (0.25, 0.6), "strong": (0.05, 0.25)}

def _resolve_strength(strength: str | float) -> float:
    if isinstance(strength, str):
        if strength not in STRENGTH_RANGES:
            raise ValueError("strength must be 'minimal', 'mild', 'medium', 'strong', or a number in [0, 1]")
        return STRENGTH_RANGES[strength]
    if isinstance(strength, (bool, np.bool_)) or not isinstance(strength, Real):
        raise TypeError("strength must be a named level or a real number")
    value = float(strength)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("numeric strength must be finite and within [0, 1]")

    # Convert individual number to a strength range
    sr_min = min(0, strength - 0.15)
    sr_max = max(1, strength + 0.15)
    return sr_min, sr_max

def save_rgb(path: Path, img: np.ndarray):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Could not encode {path}")
    encoded.tofile(str(path))  # Supports non-ASCII Windows paths


def darken(img, strength: str | float, seed: int = 1234):
    low, high = _resolve_strength(strength)

    img_tensor = img.squeeze(0)

    # Convert sRGB to linear RGB
    img_linear = kornia.color.rgb_to_linear_rgb(img_tensor)

    # Create global illumination factor to dim the image based on strength value & seed
    generator = torch.Generator(device=img_linear.device)
    generator.manual_seed(seed)
    global_factor = low + (high - low) * torch.rand(1, device=img_linear.device,
                                                      generator=generator)

    # Create a small random illumination field (32x smaller) at scale
    height, width = img_linear.shape[-2:]
    field_h = max(1, height // 32)
    field_w = max(1, width // 32)
    illumination = torch.rand(1, 1, field_h, field_w, device=img_linear.device)

    # Scale illumination field to image, field smoothens to cover image
    illumination = torch.nn.functional.interpolate(illumination, size=(height,width),
                                                   mode='bilinear', align_corners=False)
    illumination = illumination.squeeze(0) # Reverts to 3D tensor
    illumination = global_factor * (0.7 + 0.6 * illumination)
    img_dark = img_linear * illumination

    # Add simulated photon shot noise using Poisson distribution function
    photon_scale = 1000
    expected_photons = img_dark * photon_scale
    photons = torch.poisson(expected_photons)
    img_noisy = photons / photon_scale

    # Add simulated read noise (simple Gaussian noise model)
    read_noise_sigma = 0.005
    read_noise = torch.randn_like(img_noisy) * read_noise_sigma
    img_noisy += read_noise


    # Convert back to sRGB
    img_final = kornia.color.linear_rgb_to_rgb(img_noisy.clamp(0,1))

    print(f'Dimensions: {img_final.size()}')

    # Convert to numpy array
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

    strength_constant = 0.8
    result = darken(img_tensor, strength_constant, seed=6767)
    save_rgb(f'D:/_res_v2-{strength_constant}.png', result)

    # i = 1
    # for s in (STRENGTH_RANGES):
    #     result = darken(img_tensor, s, seed=6767)
    #     save_rgb(f'D:/_res_v2-{i}-{s}.png', result)
    #     i += 1




if __name__ == "__main__":
    main()