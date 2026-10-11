"""Deterministic over-exposure of uint8 RGB images (Issue #41).

This module works on image arrays and does not load images or labels.
The presets were checked on the documented day/night reference frames.
See README.md for the reference-set saturation floor and its scope; these
parameters do not guarantee saturation on every image.
"""

from numbers import Real

import numpy as np


# All transform parameters live here; see README.md for reference-set checks.
STRENGTH_LEVELS = {"mild": 0.35, "strong": 1.0}
MIN_GAMMA = 0.4
MAX_BRIGHTNESS_OFFSET = 0.6  # Fraction of the full [0, 1] intensity range.


def resolve_strength(strength: str | float) -> float:
    """Turn a named level ("mild"/"strong") into its number; pass through a value in [0, 1].

    Shared by both evaluation degradations (dark and overexposed), so "strong" and 1.0 describe
    the same benchmark.
    """
    if isinstance(strength, str):
        if strength not in STRENGTH_LEVELS:
            raise ValueError("strength must be 'mild', 'strong', or a number in [0, 1]")
        return STRENGTH_LEVELS[strength]
    if isinstance(strength, (bool, np.bool_)) or not isinstance(strength, Real):
        raise TypeError("strength must be a named level or a real number")
    value = float(strength)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("numeric strength must be finite and within [0, 1]")
    return value


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
