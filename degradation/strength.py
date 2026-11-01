from numbers import Real

import numpy as np

STRENGTH_LEVELS = {"mild": 0.35, "strong": 1.0}


def resolve_strength(strength: str | float) -> float:
    """Turn a named level ("mild"/"strong") into its number; pass through a value in [0, 1]."""
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
