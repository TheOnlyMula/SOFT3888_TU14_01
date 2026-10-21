"""Generate clean/mild/strong RGB examples and descriptive pixel statistics.

This utility does not judge acceptance criteria or enforce saturation targets.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from degradation import overexposure
from degradation.strength import STRENGTH_LEVELS


def measurements(image):
    return {
        "mean_rgb": float(image.mean()),
        "fully_saturated_percent": float(np.all(image == 255, axis=2).mean() * 100),
        "any_channel_saturated_percent": float(np.any(image == 255, axis=2).mean() * 100),
    }


def save_rgb(path, image):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Could not encode {path}")
    encoded.tofile(str(path))  # Supports non-ASCII Windows paths.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="An 8-bit, three-channel RGB image file")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; existing folders are not overwritten")
    args = parser.parse_args()
    image_bytes = args.image.read_bytes()
    bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Could not decode image: {args.image}")
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Input must be an 8-bit image with three colour channels")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    variants = {"clean": rgb}
    for level in ("mild", "strong"):
        variants[level] = overexposure.overexpose(rgb, level)
    stats = {level: measurements(image) for level, image in variants.items()}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    panels = []
    for level, image in variants.items():
        save_rgb(args.output_dir / f"{level}.png", image)
        header = np.full((64, image.shape[1], 3), 28, dtype=np.uint8)
        cv2.putText(header, level.upper(), (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)
        row = stats[level]
        caption = (f"Mean RGB: {row['mean_rgb']:.2f} | "
                   f"Fully saturated: {row['fully_saturated_percent']:.2f}%")
        cv2.putText(header, caption, (12, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (230, 230, 230), 1, cv2.LINE_AA)
        panels.append(np.vstack((header, image)))
        print(f"{level}: mean RGB = {row['mean_rgb']:.4f}, "
              f"fully saturated pixels = {row['fully_saturated_percent']:.4f}%")
    save_rgb(args.output_dir / "comparison.png", np.hstack(panels))

    report = {
        "status": "examples_generated",
        "scope": "Descriptive measurements only; no automated acceptance checks",
        "image_name": args.image.name,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "shape": list(rgb.shape),
        "module_sha256": hashlib.sha256(Path(overexposure.__file__).read_bytes()).hexdigest(),
        "strength_levels": STRENGTH_LEVELS,
        "min_gamma": overexposure.MIN_GAMMA,
        "max_brightness_offset": overexposure.MAX_BRIGHTNESS_OFFSET,
        "fully_saturated_definition": "All three RGB channels equal 255; denominator is every pixel",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
        "metrics": stats,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved example images and measurements to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
