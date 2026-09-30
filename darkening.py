from numbers import Real
import numpy as np
import torch
import kornia

STRENGTH_RANGES = {"minimal": (0.8, 0.95), "mild": (0.6, 0.8), "medium": (0.25, 0.6),
                   "strong": (0.05, 0.25)}

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
    if strength == 1:
        return 1, 1
    elif strength == 0:
        return 0, 0
    
    sr_min = max(0, strength - 0.15)
    sr_max = min(1, strength + 0.15)
    return sr_min, sr_max


def darken(img: np.ndarray, strength: str | float, seed: int = 0) -> dict:
    low, high = _resolve_strength(strength)
    img_tensor = torch.from_numpy(img) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)

    if (low == 0 and high == 0): # If strength bounds are 0, return completely black image
        return {'image': np.zeros_like(img_tensor),
                'global_illum_factor': 0,
                'range': [0,0]}

    elif (low == 1 and high == 1): # If strength bounds are 1, no change to the image
        img_final = img
        global_factor = 1
    
    else:
        # Convert sRGB to linear RGB
        img_linear = kornia.color.rgb_to_linear_rgb(img_tensor)

        # Create global illumination factor to dim the image based on strength value & seed,
        # bounded by the ranges as determined by STRENGTH_RANGES or [x - 0.15, x + 0.15) if strength constant x is inputted
        generator = torch.Generator(device=img_linear.device)
        generator.manual_seed(seed)
        global_factor = low + (high - low) * torch.rand(1, device=img_linear.device,
                                                        generator=generator)

        # Create a small random illumination field (32x smaller) at scale
        height, width = img_linear.shape[-2:]
        field_h = max(1, height // 32)
        field_w = max(1, width // 32)
        illumination = torch.rand(1, 1, field_h, field_w, device=img_linear.device,
                                generator=generator)

        # Scale illumination field to image, field smoothens to cover image
        illumination = torch.nn.functional.interpolate(illumination, size=(height,width),
                                                    mode='bilinear', align_corners=False)
        illumination = illumination.squeeze(0) # Reverts to 3D tensor
        illumination = global_factor * (0.7 + 0.6 * illumination)
        img_dark = img_linear * illumination

        # Add simulated photon shot noise using Poisson distribution function
        photon_scale = 1000
        expected_photons = img_dark * photon_scale
        photons = torch.poisson(expected_photons, generator=generator)
        img_noisy = photons / photon_scale

        # Add simulated read noise (simple Gaussian noise model)
        read_noise_sigma = 0.005
        read_noise = torch.randn_like(img_noisy, device=img_noisy.device,
                                    generator=generator) * read_noise_sigma
        img_noisy += read_noise

        # Convert back to sRGB
        img_final = kornia.color.linear_rgb_to_rgb(img_noisy.clamp(0,1))

    # Convert to numpy array
    arr_final = img_final.detach().cpu().permute(1, 2, 0).numpy()
    arr_final = (arr_final * 255).clip(0, 255).astype(np.uint8)

    max_brightness = img_dark.max().item()

    return {'image': arr_final,
            'global_illum_factor': global_factor.item(),
            'max_brightness': max_brightness}