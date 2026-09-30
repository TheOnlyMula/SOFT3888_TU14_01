"""Generate minimal/mild/medium/strong RGB examples and descriptive pixel statistics.

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
import darkening

def parse_args():
    parser = argparse.ArgumentParser(description = 'Generate a synthetic low-light image.')
    
    parser.add_argument('--image', type=Path, required=True, help='Path to an 8-bit, 3-channel RGB image as input')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory where the output image will be saved')
    parser.add_argument('--seed', type=int, default=0, help='Reproducible integer seed dictating noise generation & illumination map creation; 0 by default')

    return parser.parse_args()


def save_rgb(path, image):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Could not encode {path}")
    encoded.tofile(str(path))  # Supports non-ASCII Windows paths.


def main():
    args = parse_args()
    image_bytes = args.image.read_bytes()
    bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if bgr is None:
        raise ValueError(f"Could not decode image: {args.image}")
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Input must be an 8-bit image with three colour channels")
    rgb_input = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    variants = {"clean":
                {'image': rgb_input, 'global_illum_factor': 1, 'max_brightness': 1}}
    for level in darkening.STRENGTH_RANGES:
        variants[level] = darkening.darken(rgb_input, level, seed = args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=False)

    panels = []
    for level in variants.keys():
        results_dict = variants[level]
        image = results_dict['image']
        illum_factor = results_dict['global_illum_factor']
        max_brightness = results_dict['max_brightness']

        save_rgb(args.output_dir / f"{level}.png", image)
        header = np.full((64, image.shape[1], 3), 28, dtype=np.uint8)

        if level == 'clean':
            cv2.putText(header, level.upper(), (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(header, f'{level.upper()} [{darkening.STRENGTH_RANGES[level][0]}, {darkening.STRENGTH_RANGES[level][1]})', (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (255, 255, 255), 1, cv2.LINE_AA)

        caption = (f"Global illumination factor: {illum_factor:.4f} | "
                   f"Peak brightness: {max_brightness:.4f}")
        cv2.putText(header, caption, (12, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (230, 230, 230), 1, cv2.LINE_AA)
        panels.append(np.vstack((header, image)))
    save_rgb(args.output_dir / "comparison.png", np.hstack(panels))

    report = {
        "status": "examples_generated",
        "scope": "Descriptive measurements only; no automated acceptance checks",
        "image_name": args.image.name,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "shape": list(rgb_input.shape),
        "module_sha256": hashlib.sha256(Path(darkening.__file__).read_bytes()).hexdigest(),
        "strength_levels": darkening.STRENGTH_RANGES,
        "fully_darkened_definition": "All three RGB channels equal 0; denominator is every pixel",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved example images and measurements to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
