# Darkening reference guide (Issue #40)

[Back to the project README](../../README.md)

`darkening.py` provides `darken(image, strength, seed = 0)` for a non-empty `H x W x 3` NumPy RGB array with dtype `uint8`. The seed shall be 0 if not manually provided. It returns a dictionary containg the resultant NumPy array alongside seed, illumination factor, and raw mean brightness:
```python
{'image': np.ndarray, 'seed': int, 'mean_brightness': float}
```
OpenCV BGR-to-RGB conversion is performed on `image` before applying tensor normalization or event concatenation. This standalone function does not change the existing training/evaluation pipeline or touch labels.

```python
from darkening import darkening

minimal_rgb = darken(rgb, "minimal", seed = 1234)
mild_rgb = darken(rgb, "mild", seed = 1234)
medium_rgb = darken(rgb, "medium", seed = 1234)
strong_rgb = darken(rgb, "strong", seed = 1234)
custom_rgb = darken(rgb, 0.6)  # Continuous strength in [0, 1].
```

Strength is either a named strength level (see below), or a finite real number in `[0, 1]`.
Invalid array types/dtypes and invalid strength types raise `TypeError`;
invalid shapes, empty arrays, unknown presets and out-of-range or non-finite
strength values raise `ValueError`.

| Strength level | Global illumination factor range |
| --- | ---: |
| minimal | [0.7, 0.9) |
| mild | [0.5, 0.7) |
| medium | [0.3, 0.5) |
| strong | [0.1, 0.3) |

A custom strength level `i` will generate a global illumination factor range of [`i` - 0.1, `i` + 0.1], bounded by 0 and 1, from which the global illuimnation factor will be sampled to prevent overfitting. 
```python
generator = torch.Generator(device=img_linear.device)
            generator.manual_seed(seed)
            global_factor = low + (high-low) * torch.rand(1, device=img_linear.device, generator=generator)
```
Note that `i` = 0 will generate a pitch-black image, and `i` = 1 will result in no transformation to the image.

## Generate examples and measurements

Run the exporter from the repository root with an original 8-bit RGB image and a new output directory. It requires NumPy and OpenCV, but no model. The resultant image will be created in the same device as the original image.

```bash
python tools/generate_darkened_examples.py --image "/path/to/DSEC_ROOT/train/train/interlaken_00_c/images/left/distorted/000003.png" --output-dir "runs/issue41_examples/day_1" -- seed 1234

python tools/generate_darkened_examples.py --image "/path/to/DSEC_ROOT/train/zurich_city_09_a/images/left/distorted/000005.png" --output-dir "runs/issue40_examples/night_1" --seed 1234
```

Each output folder contains `clean.png`, `minimal.png`, `mild.png`, `medium.png`, `strong.png`,
`comparison.png` and `metrics.json`. The JSON records the source-image hash,
parameters, runtime versions, and numerical stats in the resultant dictionary (i.e., seed, mean brightness, and percentage of discernably dark pixels). Input images are read without changing them; existing output folders are not overwritten. Non-ASCII Windows paths are supported.
