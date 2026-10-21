import numpy as np
import torch
import kornia

from .strength import resolve_strength

STRONG_ILLUMINATION = 0.2  # Illumination factor at strength 1
VARIATION = 0.1  # Each image's global illumination factor is drawn within +/- this of its level
PHOTON_SCALE = 1000
READ_NOISE_SIGMA = 0.005
DARK_THRESHOLD = 0.01  # Linear light at or below which a pixel counts as dark


def darken(img: np.ndarray, strength: str | float, seed: int = 0) -> dict:
    """Darken an H x W x 3 uint8 RGB image; `seed` fixes the illumination field and the noise.

    `strength` is "mild", "strong" or a number in [0, 1]. Returns the darkened image, the global
    illumination factor drawn for it and, before noise, its mean linear brightness and the
    proportion of pixels that are dark in every channel.
    """
    strength = resolve_strength(strength)
    img_linear = kornia.color.rgb_to_linear_rgb(torch.from_numpy(img).permute(2, 0, 1) / 255.0)

    if strength == 0:
        global_factor = torch.ones(1)
        img_dark = img_linear
        image = img.copy()
    else:
        illumination = 1.0 - (1.0 - STRONG_ILLUMINATION) * strength
        low, high = max(0, illumination - VARIATION), min(1, illumination + VARIATION)
        generator = torch.Generator(device=img_linear.device)
        generator.manual_seed(seed)
        global_factor = low + (high - low) * torch.rand(1, device=img_linear.device, generator=generator)

        # A small random illumination field (32x smaller), smoothly upscaled to cover the image
        height, width = img_linear.shape[-2:]
        field = torch.rand(1, 1, max(1, height // 32), max(1, width // 32),
                           device=img_linear.device, generator=generator)
        field = torch.nn.functional.interpolate(field, size=(height, width),
                                                mode='bilinear', align_corners=False).squeeze(0)
        img_dark = img_linear * (global_factor * (0.7 + 0.6 * field))

        # Sensor noise: Poisson photon shot noise, then Gaussian read noise
        photons = torch.poisson(img_dark * PHOTON_SCALE, generator=generator)
        read_noise = torch.randn(photons.shape, dtype=photons.dtype, device=photons.device,
                                 generator=generator) * READ_NOISE_SIGMA
        img_noisy = (photons / PHOTON_SCALE + read_noise).clamp(0, 1)
        img_srgb = kornia.color.linear_rgb_to_rgb(img_noisy).detach().cpu().permute(1, 2, 0).numpy()
        image = (img_srgb * 255).clip(0, 255).astype(np.uint8)

    return {'image': image,
            'seed': seed,
            'global_illum_factor': global_factor.item(),
            'mean_brightness': img_dark.mean().item(),
            'proportion_dark': (img_dark <= DARK_THRESHOLD).all(dim=0).float().mean().item()}
