import numpy as np

from .strength import resolve_strength

# All transform parameters live here; see docs/overexposure/README.md for the calibration.
MIN_GAMMA = 0.4
# Fraction of the full [0, 1] intensity range. Calibrated on 400 training frames so that "strong"
# saturates a median 23% of pixels: severe, but objects stay visible (0.6 whitened 72%).
MAX_BRIGHTNESS_OFFSET = 0.4


def overexpose(image: np.ndarray, strength: str | float) -> np.ndarray:
    """Return a new uint8 RGB image with deterministic highlight clipping.

    Args:
        image: A non-empty H x W x 3 RGB array with dtype uint8. Convert
            OpenCV BGR input to RGB before calling this function.
        strength: 'mild', 'strong', or a continuous value in [0, 1].
            Zero returns an independent, unchanged copy of the input.

    The transform is applied independently to each channel:
        x = image / 255
        gamma = 1 - (1 - MIN_GAMMA) * strength
        offset = MAX_BRIGHTNESS_OFFSET * strength
        output = round(255 * clip(x ** gamma + offset, 0, 1))

    Gamma brightens midtones, while the additive offset causes highlights
    to exceed the valid range and clip. Shape is preserved; the input is
    never modified and no random numbers are used. This is an illumination
    degradation operator, not a geometric transform or a physical glare model.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be non-empty")

    value = resolve_strength(strength)
    if value == 0.0:
        return image.copy()

    gamma = 1.0 - (1.0 - MIN_GAMMA) * value
    offset = MAX_BRIGHTNESS_OFFSET * value
    normalized = image.astype(np.float64) / 255.0
    exposed = np.power(normalized, gamma) + offset
    return np.rint(np.clip(exposed, 0.0, 1.0) * 255.0).astype(np.uint8)
