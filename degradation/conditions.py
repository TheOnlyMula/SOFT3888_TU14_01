import hashlib
import math
from numbers import Integral

import numpy as np

from .darkening import darken
from .overexposure import overexpose
from .strength import STRENGTH_LEVELS

DEFAULT_DEGRADATION_SHARES = (0.5, 0.25, 0.25)
DEFAULT_STRENGTH_RANGE = (STRENGTH_LEVELS["mild"], STRENGTH_LEVELS["strong"])
DEFAULT_AUGMENTATION_SEED = 0


def validate_augmentation_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("augmentation seed must be a non-negative integer")
    return int(seed)


def validate_degradation_shares(shares):
    """Validate probabilities ordered as clean, dark, overexposed."""
    shares = tuple(float(value) for value in shares)
    if (len(shares) != 3 or any(not math.isfinite(p) or p < 0 for p in shares)
            or not math.isclose(sum(shares), 1.0, rel_tol=0, abs_tol=1e-8)):
        raise ValueError("degradation shares must be three finite non-negative values summing to 1")
    return shares


def validate_strength_range(strength_range):
    """Validate an inclusive minimum/maximum on the benchmark's [0, 1] scale."""
    values = tuple(float(value) for value in strength_range)
    if (len(values) != 2 or any(not math.isfinite(value) for value in values)
            or not 0 <= values[0] <= values[1] <= 1):
        raise ValueError("strength range must be two finite values with 0 <= min <= max <= 1")
    return values


def augment_rgb_degradation(
    image, shares,
    dark_strength_range=DEFAULT_STRENGTH_RANGE,
    overexposed_strength_range=DEFAULT_STRENGTH_RANGE,
    *, rng: np.random.Generator,
):
    """Sample a fresh condition and uniform degradation strength on each access."""
    condition = rng.choice(("clean", "dark", "overexposed"), p=shares)
    if condition == "dark":
        strength = rng.uniform(*dark_strength_range)
        return darken(image, strength=strength, seed=int(rng.integers(2**63)))["image"]
    if condition == "overexposed":
        strength = rng.uniform(*overexposed_strength_range)
        return overexpose(image, strength)
    return image


def degradation_seed(
    evaluation_seed: int,
    sequence_name: str,
    frame_index: int,
    degradation: str,
    strength: float,
) -> int:
    seed_input = (
        f"{evaluation_seed}|{sequence_name}|{frame_index}|{degradation}|{strength}"
    ).encode("utf-8")

    digest = hashlib.sha256(seed_input).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)
