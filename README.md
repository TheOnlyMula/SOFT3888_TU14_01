# SOFT3888_TU14_01

This project will develop an event-privileged training framework using the DSEC-Detection dataset. A teacher model will use paired RGB and event-camera information to perform traffic object detection, while an RGB-only student model will learn from the teacher through knowledge distillation. This approach will investigate whether object-aware distillation can improve the transfer of information from the teacher to the student. The resulting student model will operate independently using RGB input only during inference.

Our capstone project for SOFT3888 aims to address the drawbacks of RGB cameras in object detection by integrating data from event cameras into model training. If successful, the primary benefit of this approach is detecting objects in traffic more accurately with reduced computational overhead in a more lightweight framework. This allows for faster response times while maintaining accuracy in adequate lighting conditions, and surpassing the accuracy of conventional RGB systems in harsh lighting conditions.

## Repository structure

```text
SOFT3888_TU14_01/
├── README.md
├── dataset_loader.py
├── training.py
├── distillation.py
├── train_distilled_student.py
├── train_rgb_student.py
├── train_teacher.py
├── evaluate.py
├── utils.py
├── log.py
├── environment.yml
├── yolo26n.pt
├── DSEC_ROOT/
│   └── train/
│       └── ... labelled DSEC sequences ...
├── DSEC-det/
│   └── official DSEC detection utilities
├── runs/
│   └── training outputs and logs
├── tests/
│   └── regression and smoke tests
├── tools/
│   └── utilities for exports, diagnostics, and smoke checks
└── .gitignore
```

## Core components

### dataset_loader.py

This file defines the DSEC dataset access layer. It creates a custom dataset that:

- reads RGB images from DSEC sequences,
- optionally reads event tensors from event files,
- converts event polarity information into a 2-channel histogram,
- maps labels into YOLO-style bounding box targets,
- supports both RGB-only and fused RGB + event training paths.

Images and labels come from the DSEC detector utilities. For real-time detection, each RGB image is paired only with events from the 50 ms _before_ its timestamp (`load_causal_events` in `dataset_loader.py`), so the model never sees events that happen after the image. DSECDet's own `sync` windows are not used for events: `sync="back"` returns the 50 ms after the image, and `sync="front"` pairs the image with the previous frame's labels. The first frame of each sequence has no preceding events, so its event channels are empty.

### training.py

The shared training pipeline used by both models. It:

- builds a DSEC RGB dataloader for train/validation splits,
- initialises a YOLO model from the pretrained `yolo26n.pt` checkpoint,
- trains with a YOLO-style loss,
- logs training metrics,
- saves the best validation checkpoint,
- optionally runs a final test evaluation.

### train_distilled_student.py

Trains the RGB student with unweighted feature distillation from a frozen RGB + event teacher (`distillation.py`). Training batches carry RGB + event channels: the teacher sees all of them, the student only the RGB ones. Validation and test stay RGB-only, so student results never depend on event data.

```bash
python train_distilled_student.py   --dataset-path dataset   --epochs 50   --teacher-checkpoint runs/teacher/exp_xxx/best_teacher_yolo.pt   --lambda-distill 1.0
```

The distillation loss is the mean smooth L1 distance between the student's and the frozen teacher's one-to-one neck features, applied at every spatial location (no object mask and no event-quality weight yet). Each epoch logs `distill_loss` and its weighted contribution to the total loss.

### train_rgb_student.py and train_teacher.py

Thin entry points into `training.py`. The two models are trained identically; the only difference is the input:

- `train_rgb_student.py`: 3 channels (RGB).
- `train_teacher.py`: 5 channels (RGB + 2-channel event histogram).

Both accept the same arguments (`--dataset-path`, `--epochs`, `--batch-size`, `--num-workers`, `--max-samples`, `--skip-test`, `--console-level`).

### evaluate.py

This file holds the evaluation used for both per-epoch validation and the final test step, so validation and test numbers are computed identically. In a single pass over the data it:

- computes the per-image loss (box, class, L1),
- computes precision, recall, mAP@0.5, mAP@0.5:0.95 and class-agnostic mAP@0.5 (all classes merged, to separate missed objects from wrong classes),
- reports per-class results,
- for the test step, restores the best checkpoint and saves the confusion matrix, PR curves and example predictions to `runs/<role>/exp_NNN/test/`.

### utils.py

This module contains shared utility functions for:

- selecting the correct compute device,
- extracting trainable layers from the YOLO model,
- creating the optimizer,
- loading a checkpoint,
- reading the best saved checkpoint state.

## Environment setup

The project expects a Python environment with PyTorch, Ultralytics, OpenCV, and the DSEC dataset utilities. The repository includes an environment file.

### 1. Clone the repository

```bash
git clone <repo-url>
cd SOFT3888_TU14_01
```

### 2. Create the environment

```bash
mamba env create -f environment.yml
mamba activate dsec-env
```

If you are using conda instead of mamba, the same environment definition should work with `conda env create -f environment.yml`.

For macOS and Linux systems, you may also use a python env using the following process:

```bash
python -m venv .venv
source .venv/bin/activate
```

This will isolate the current terminal session and run a Python virtual environment.

### 3. Install the DSEC detection utilities

The repo depends on the external `DSEC-det` package for dataset parsing and labelling support.

```bash
git clone https://github.com/uzh-rpg/DSEC-det.git
cd DSEC-det
pip install -e .
```

Make sure the environment remains active while running the training scripts.

## Data layout

This project expects the DSEC dataset to be available under a root folder such as `DSEC_ROOT`.

```text
DSEC_ROOT/
└── train/
    ├── sequence_01/
    │   ├── calibration/
    │   ├── events/
    │   ├── images/
    │   └── object_detections/
    └── sequence_02/
        ├── calibration/
        ├── events/
        ├── images/
        └── object_detections/

```

For the project to work correctly, each sequence should contain the usual DSEC structure, including:

- `images/left/distorted/` for RGB frames
- `events/` for event files
- `object_detections/` with tracking label files
- `timestamps.txt` and corresponding metadata for alignment

The dataset loader uses the DSEC split configuration in `DSEC-det/config/train_val_test_split.yaml` to determine which sequences belong to train, validation, and test subsets.

## Training pipeline

### Student RGB-only training

Run the student model training from the project root:

```bash
python train_rgb_student.py \
  --dataset-path "/path/to/DSEC_ROOT" \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 1
```

Example with validation-only setup and no final test evaluation:

```bash
python train_rgb_student.py \
  --dataset-path "/path/to/DSEC_ROOT" \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 10 \
  --skip-test
```

The script does the following:

- loads the DSEC RGB-only train and validation sets,
- shuffles the training data,
- collates labels into a YOLO-compatible target structure,
- clips gradients to avoid exploding updates,
- validates after each epoch,
- saves the model with the lowest validation loss,
- runs a final test evaluation unless `--skip-test` is provided.

Enable a random degradation mix for RGB-only student training:

```bash
python train_rgb_student.py \
  --dataset-path "/path/to/DSEC_ROOT" \
  --epochs 10 \
  --degradation-augmentation \
  --degradation-shares 0.5 0.25 0.25 \
  --dark-strength-range 0.35 1.0 \
  --overexposed-strength-range 0.35 1.0 \
  --augmentation-seed 42
```

Augmentation is off by default. Shares are probabilities in clean, darkened,
over-exposed order; they must be non-negative and sum to 1. The default is
50%/25%/25%. Each training image independently draws a fresh condition on every
load, so batch and epoch proportions fluctuate. Augmentation affects only the
training loader: validation (including checkpoint selection) and final test
evaluation use clean RGB images, identical to those used without augmentation.
To evaluate degraded images, explicitly run the [degraded evaluation benchmark](#degraded-evaluation-benchmark).
For each degraded image, strength is drawn uniformly
from the selected condition's range. Both ranges default to the benchmark's
`mild` (0.35) through `strong` (1.0) levels. Bounds must satisfy
`0 <= MIN <= MAX <= 1`; equal bounds give a fixed strength. Clean images do not
draw a strength. Darkening still uses the existing temporary darkening function.
These options are for the RGB-only student without a teacher checkpoint and do
not change benchmark settings.

`--augmentation-seed` (default `0`, non-negative integer) controls a dedicated
random stream for condition selection, strength and darkening noise. It is
independent of the benchmark's `--evaluation-seed` and global NumPy/PyTorch RNGs.
Each loader worker has its own stream, which advances across epochs. The same
seed reproduces the augmentation sequence for the same sample order, batch size
and worker count; this seed does not control training shuffle or model randomness.
Changing the worker count can change the augmentation sequence. Augmentation RNG
state is not checkpointed, so restarting a loader restarts its streams.

Every run's `runs/student/exp_XXX/config.yaml` records `Training Augmentation`:
the enabled flag, named condition shares, both strength ranges and the seed,
including default values when options are omitted.

### Teacher RGB + event training

The teacher runs exactly the same pipeline as the student (training, validation, test loss and mAP), but its input tensor also contains the event histogram.

```bash
python train_teacher.py \
  --dataset-path "/path/to/DSEC_ROOT" \
  --batch-size 4 \
  --num-workers 0 \
  --epochs 1
```

This path uses the event histogram generated from event polarity, stacking it alongside RGB features before feeding the model.

## Evaluation workflow

The evaluation code is built around the YOLO detection model and uses DSEC test data. The repository saves training logs under `runs/` and stores checkpoints alongside the training logs.

### Validation metric logic

During training, validation is performed using a model in evaluation mode and `torch.no_grad()`. The script computes:

- average validation batch loss,
- average validation box loss,
- average validation classification loss,
- average validation L1 loss.

The best checkpoint is selected using the lowest validation batch loss.

### Test evaluation command

```bash
python evaluate.py \
  --weights "/path/to/checkpoint.pt" \
  --type student \
  --dataset-path "/path/to/DSEC_ROOT" \
  --batch-size 16 \
  --num-workers 8
```

Both models use the stock Ultralytics YOLO26n architecture with an 8-class DSEC head (built by `build_model` in `utils.py`). The `--type` argument sets the number of input channels:

- `student` -> 3 channels (RGB)
- `teacher` -> 5 channels (RGB + 2 event histogram channels)

The evaluation loads the saved checkpoint and evaluates it on the test split with the same function used during training, writing metrics to the log and plots to the run's `test/` folder.

### Degraded evaluation benchmark

The degraded evaluation benchmark compares a trained checkpoint on the same test frames under three conditions:

- clean
- darkened
- over-exposed

Run the benchmark with:

```bash
PYTHONPATH=. python evaluate.py \
  --weights "/path/to/checkpoint.pt" \
  --type student \
  --dataset-path "/path/to/DSEC_ROOT" \
  --benchmark \
  --degradation-strength strong \
  --evaluation-seed 42
```

The benchmark reports mAP@0.5, mAP@0.5:0.95, recall, and the performance drop from the clean condition.

`--degradation-strength` accepts the named levels `mild` (0.35) and `strong` (1.0) defined in `overexposure.py`, or a number in [0, 1]. The same value is used for the darkened and over-exposed conditions, and over-exposure is applied by `overexposure.overexpose`.

The evaluation seed keeps benchmark inputs reproducible. Each condition also produces an RGB fingerprint, which can be compared between runs to confirm that the same input images were evaluated.

For fair comparisons between models, use the same dataset split, degradation settings, evaluation seed, and benchmark configuration.

### Event representation

The teacher's event channels are a 2-channel positive/negative histogram over the causal window, log-scaled and normalised into [0, 1]:

```
channel = min( log1p(count) / log1p(EVENT_SCALE_K), 1 )
```

`EVENT_SCALE_K = 44` is the 99.9th percentile of non-zero per-pixel event counts, measured on the training split only with `tools/measure_event_scale.py`. Log scaling keeps low counts distinguishable, since a typical active pixel sees only ~2 events, and the divisor puts the event channels on the same [0, 1] scale as RGB. The constant is frozen: teacher and distillation runs must use the same value, and changing it requires retraining the teacher.

## Dataset and training notes

- The project uses a deterministic split based on the official DSEC configuration file.
- Training and validation are both performed on labelled subsets, while test evaluation is reserved for final measurement.
- The code expects a valid DSEC root path with all required subdirectories aligned to the same sequence ID.
- On Windows, prefer ASCII-only dataset paths when possible because OpenCV and some file-handling utilities can sometimes fail on non-ASCII characters.
- The training pipeline saves both a best checkpoint and a latest checkpoint, so progress can resume if a run is interrupted.

## Testing and validation

Run the unit and integration checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

The test suite covers the project’s expected data-loading and training workflow behaviour.

## Smoke checks and diagnostics

The repository also includes helper workflows for quick integration verification. For example:

```powershell
.\.venv\Scripts\python.exe tools/run_issue17_smoke.py --dataset-root "D:\DSEC_ROOT" --output-dir "D:\DSEC_issue17_verify"
```

This test is designed to validate a reduced subset of the pipeline without requiring a full training run across the whole dataset.

## Expected outputs

Typical outputs include:

- training logs in `runs/`
- best validation checkpoints such as `best_student_yolo.pt`
- latest checkpoints for resume support
- evaluation logs that report the loss values used for model selection
- test results generated by the evaluation pipeline

## Troubleshooting

If training or evaluation fails, check the following first:

1. Ensure the dataset path points to the correct DSEC root.
2. Confirm that `DSEC-det` is installed in the active environment.
3. Confirm `train/val/test` directories are populated and labelled correctly.
4. Ensure the checkpoint path and YAML config path are valid.
5. Check that the environment matches the required Python and dependency versions.

This project is meant to serve as a full training and evaluation workflow for DSEC-based RGB and RGB + event detection experiments. It is especially useful for experimenting with student–teacher knowledge transfer and multi-modal perception pipelines in a structured and reproducible setup.

## Deterministic overexposure (Issue #41)

`overexpose(image, strength)` returns a new, same-shape `uint8` RGB array
without randomness or input mutation. Input is a non-empty `H x W x 3` array;
convert BGR to RGB and apply it before tensor normalization or event concatenation.

```python
from overexposure import overexpose
mild_rgb = overexpose(rgb, "mild")
strong_rgb = overexpose(rgb, "strong")
custom_rgb = overexpose(rgb, 0.6)  # Continuous strength in [0, 1].
```

| Preset | Strength | Gamma | Normalized offset |
| --- | ---: | ---: | ---: |
| mild | 0.35 | 0.79 | 0.21 |
| strong | 1.00 | 0.40 | 0.60 |

The proposed `strong` target is **at least 2% fully saturated pixels** (all RGB
channels equal 255) on `zurich_city_09_a` frames `000007`, `000100`, `000200`
and `000300`. This target applies only to these calibration frames.

[Day examples](docs/overexposure/day_000007.png) and
[night examples](docs/overexposure/night_000300.png) show clean/mild/strong.
See the [detailed guide](docs/overexposure/README.md) for the formula and measured results.

Generate examples and descriptive measurements into a new output directory:

```bash
python tools/generate_overexposure_examples.py --image "/path/to/rgb.png" --output-dir "runs/issue41_examples/sample_01"
```
