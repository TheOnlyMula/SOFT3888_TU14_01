# SOFT3888_TU14_01

This project will develop an event-privileged training framework using the DSEC-Detection dataset. A teacher model will use paired RGB and event-camera information to perform traffic object detection, while an RGB-only student model will learn from the teacher through knowledge distillation. This approach will investigate whether object-aware distillation can improve the transfer of information from the teacher to the student. The resulting student model will operate independently using RGB input only during inference.

Our capstone project for SOFT3888 aims to address the drawbacks of RGB cameras in object detection by integrating data from event cameras into model training. If successful, the primary benefit of this approach is detecting objects in traffic more accurately with reduced computational overhead in a more lightweight framework. This allows for faster response times while maintaining accuracy in adequate lighting conditions, and surpassing the accuracy of conventional RGB systems in harsh lighting conditions.

## Repository structure

```text
SOFT3888_TU14_01/
├── README.md
├── environment.yml
├── .gitignore
├── dataset_loader.py
├── augment.py
├── darkening.py
├── overexposure.py
├── distillation.py
├── training/
│   ├── __init__.py
│   ├── cli.py
│   ├── run.py
│   ├── loop.py
│   ├── losses.py
│   ├── schedule.py
│   └── reporting.py
├── train_distilled_student.py
├── train_rgb_student.py
├── train_teacher.py
├── evaluate.py
├── benchmark_reporting.py
├── overfit.py
├── utils.py
├── log.py
├── docs/
│   └── event_scale.json, darkening and overexposure guides
├── tests/
│   └── regression and smoke tests
├── tools/
│   └── utilities for exports, diagnostics, and smoke checks
│
│   Not in Git (created during setup, see "Environment setup"):
├── DSEC-det/        official DSEC detection utilities, cloned into the project root
├── yolo26n.pt       pretrained weights, downloaded automatically on the first run
├── runs/            training outputs and logs
└── DSEC_ROOT/       the dataset; may live anywhere, passed with --dataset-path
    ├── train/       training and validation sequences
    └── test/        test sequences
```

## Core components

### dataset_loader.py

This file defines the DSEC dataset access layer. It creates a custom dataset that:

- reads RGB images from DSEC sequences,
- optionally reads event tensors from event files,
- converts event polarity information into a 2-channel histogram,
- maps labels into YOLO-style bounding box targets,
- supports both RGB-only and fused RGB + event training paths,
- augments the training split when `augment=True` (see below).

Images and labels come from the DSEC detector utilities. For real-time detection, each RGB image is paired only with events from the 50 ms _before_ its timestamp (`load_causal_events` in `dataset_loader.py`), so the model never sees events that happen after the image. DSECDet's own `sync` windows are not used for events: `sync="back"` returns the 50 ms after the image, and `sync="front"` pairs the image with the previous frame's labels. The first frame of each sequence has no preceding events, so its event channels are empty.

### augment.py

Training-time augmentation, used by every model so student comparisons stay matched. One random draw moves the RGB pixels, the event pixels and the boxes together, so events stay aligned with the RGB the student sees:

- horizontal flip (probability 0.5),
- zoom of 0.5-1.5x about the centre and a shift of up to 10% of the width and height,
- hue shift and saturation jitter on the RGB channels only (hue moves by the same amount for every colour, as in Ultralytics 8.4). Brightness is left alone because brightness shifts belong to the degradation-augmentation baseline.

Pixels uncovered by the zoom or shift are grey (114) in RGB and 0 in the event channels, since any other event value would invent events. Boxes that end up mostly outside the image, or cut into slivers more than 100 times longer than wide, are dropped. Every sample makes the same random draws with or without events, so for the same `--seed` the RGB student and the distilled student see identical augmented RGB.

### training/

The shared training pipeline used by both models. The entry scripts import `main` from the package; its modules follow the order a run happens in:

| Module | Contents |
| --- | --- |
| `cli.py` | `parse_args` and `--resume` handling |
| `run.py` | `main()`: builds the data, models and optimiser from the arguments, writes `config.yaml`, trains, then tests |
| `loop.py` | `train_model()`: epochs, gradient accumulation, optimiser steps, validation and best-checkpoint selection |
| `losses.py` | `batch_losses()`: one batch's detection loss and, with a teacher, its distillation loss for each `--distill-mode` |
| `schedule.py` | warmup constants, `lr_factor` and `apply_warmup` (Ultralytics' schedule) |
| `reporting.py` | each epoch's `train.log` summary, its `results.csv` row and the quality-weight statistics |

The pipeline:

- builds a DSEC RGB dataloader for train/validation splits,
- initialises a YOLO model from the pretrained `yolo26n.pt` checkpoint,
- trains with a YOLO-style loss on augmented frames (`--no-augment` turns this off),
- accumulates gradients to about 64 images per optimiser step, as Ultralytics does,
- follows Ultralytics' YOLO26 optimisation recipe: MuSGD (lr 0.01, Nesterov momentum, 3x learning rate on the detection head's classification layers, weight decay on conv and linear weights only), the summed rather than averaged loss, a 3-epoch warmup of learning rate, momentum and accumulation, then a linear decay to 1%,
- runs with deterministic GPU kernels (as Ultralytics does), which costs some speed,
- logs training metrics, and after every epoch adds them to the run's `results.csv` and redraws the training curves in `results.png` (Ultralytics' `plot_results`),
- saves the best validation checkpoint,
- optionally runs a final test evaluation.

### train_distilled_student.py

Trains the RGB student with feature distillation from a frozen RGB + event teacher (`distillation.py`). Training batches carry RGB + event channels: the teacher sees all of them, the student only the RGB ones. Validation and test stay RGB-only, so student results never depend on event data.

```bash
python train_distilled_student.py \
  --dataset-path "/path/to/DSEC_ROOT" \
  --epochs 50 \
  --teacher-checkpoint runs/teacher/exp_xxx/best_teacher_yolo.pt \
  --lambda-distill 1.0 \
  --distill-mode object

# Same run with each image's object-aware loss scaled by the quality weight w
python train_distilled_student.py \
  --dataset-path "/path/to/DSEC_ROOT" \
  --epochs 50 \
  --teacher-checkpoint runs/teacher/exp_xxx/best_teacher_yolo.pt \
  --lambda-distill 1.0 \
  --distill-mode object-quality
```

A teacher checkpoint must exist first (see [Teacher RGB + event training](#teacher-rgb--event-training)). Distilled runs are saved to `runs/distill/exp_NNN/`, with the best weights in `best_distill_yolo.pt`. Evaluate them with `evaluate.py --type student`, because the distilled student takes RGB input only.

The distillation loss is the smooth L1 distance between the student's and the frozen teacher's one-to-many P3 neck features (stride 8). Student and teacher share an architecture, so the features match without a projection layer. `--distill-mode` chooses where it is applied:

- `unweighted` (default, the baseline): every spatial location counts equally.
- `object`: only the labelled objects count (proposal eq. 2). Each image's boxes, after augmentation, are enlarged by `MASK_ENLARGEMENT` (1.2) about their centres and marked on the feature grid; the distance is averaged over those cells. Images without objects add no distillation loss. With a mask covering everything, this equals the unweighted loss, so one `--lambda-distill` serves both modes.
- `object-quality`: `object` with each image's loss multiplied by the deterministic quality weight `w = clip(s_e * c_t, 0, 1)` (proposal eq. 3), so unreliable event windows and unsure teacher outputs supervise the student less. `w` is the only difference between the two modes: with every `w = 1` they train identically.

#### Quality weight (`object-quality` mode)

For each training image `i`, `distillation.quality_weight` combines:

- `s_e`, the normalised event density: `event_density` of the teacher's 5-channel input (described below).
- `c_t`, the mean teacher confidence over its retained detections: `teacher_confidence` averages the confidence of the frozen teacher's end-to-end detections at or above `TEACHER_CONFIDENCE_THRESHOLD` (0.25). An image with no retained detection gets `c_t = 0`. The training step computes `c_t` once per batch from the same teacher forward pass used for distillation, in every distillation mode, and `object-quality` reuses it.
- `w_i = clip(s_e * c_t, 0, 1)`, which always lies in [0, 1]. An empty event window or a teacher that detects nothing confidently gives `w_i = 0`.

`w` is computed and applied only in `object-quality` mode. It is detached and has no parameters: nothing in it is learned and no gradient flows through it, so an image with `w_i = 0` sends no distillation gradient to the student (its detection loss still trains as normal). The batch distillation loss is

```
L_distill = (1 / B) * sum_i  w_i * L_distill_i
```

where `L_distill_i` is image `i`'s object-aware loss and `B` is the number of images in the batch (`distillation.quality_weighted_loss`). The mean runs over every image, including those with `w_i = 0`, so down-weighting an image lowers the batch loss rather than handing its share to the other images. This loss then enters the total loss as `lambda_distill * L_distill` (proposal eq. 1), exactly as in the other modes.

Each epoch adds the mean, minimum and maximum `w` over that epoch's training images, and the fraction of images with `w = 0`, to `train.log`:

```
  train quality_weight mean = 0.4120 min = 0.0000 max = 0.9310 zero_fraction = 0.0850
```

and to `results.csv` as `train/quality_weight_mean`, `train/quality_weight_min`, `train/quality_weight_max` and `train/quality_weight_zero_fraction`. These columns appear only in `object-quality` runs and are not drawn in `results.png`. This mode needs `docs/event_density_reference.json` (see below) and stops before training if it is missing.

The mask and the quality weight are built in the training step only and are never part of evaluation or inference. Each epoch logs `distill_loss`, the mode and the weighted contribution to the total loss; `config.yaml` records the distillation mode, the teacher confidence threshold, the mask enlargement (`object` and `object-quality`) and the event-density reference (`object-quality`).

`distillation.event_density(teacher_images)` computes a score for each frame from the same 5-channel input tensor passed to the teacher, whose event channels contain the causal 50 ms histogram. Coverage uses the whole frame: a pixel is active if either event polarity is positive, and each pixel counts once. The score is `clamp(active_pixels / (height * width) / reference, 0, 1)`. The fixed reference is loaded once per process from `docs/event_density_reference.json`; an explicit `reference` argument can also be supplied. It is separate from the per-pixel count scale `EVENT_SCALE_K`. No events gives 0; coverage at or above the reference gives 1. This score is `s_e` in the `object-quality` quality weight.

Before final training, run `python tools/measure_event_density.py --dataset-path /path/to/DSEC` once. The tool creates `docs/event_density_reference.json` locally; this generated file is ignored by Git and should be retained with the training artifacts, rather than committed. The script requires all 41 configured training sequences, reads every training frame without augmentation, and uses the 90th percentile of whole-frame coverage (including empty frames), with linear percentile interpolation. It rejects validation/test splits and refuses to overwrite an existing JSON. Training and testing reuse the frozen reference; they never regenerate it. No reference config is supplied in this change. Boundary tests are deferred to a separate task.

### train_rgb_student.py and train_teacher.py

Thin entry points into the `training` package. The two models are trained identically; the only difference is the input:

- `train_rgb_student.py`: 3 channels (RGB).
- `train_teacher.py`: 5 channels (RGB + 2-channel event histogram).

Both accept the same arguments (`--dataset-path`, `--epochs`, `--batch-size`, `--num-workers`, `--max-samples`, `--skip-test`, `--seed`, `--no-augment`, `--resume`, `--console-level`). Use the same `--seed` for every student you compare; it fixes model initialisation, the shuffle order and the augmentation draws.

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

The project needs Python 3.11 (3.10 or newer works), PyTorch, Ultralytics 8.4.128, OpenCV, h5py with `hdf5plugin`, and the DSEC detection utilities. Run every command below from the project root unless a step says otherwise.

### Prerequisites

- Git, and either mamba/conda or Python 3.11 with `venv`.
- `wget` and `unzip` for the dataset download scripts. macOS does not ship `wget`; install it with `brew install wget`.
- An internet connection on the first training run, so Ultralytics can download `yolo26n.pt`.
- An NVIDIA GPU with CUDA for full training runs. Without one, the code falls back to the CPU (including on Apple Silicon Macs). That is enough for the tests and the short smoke run below, but too slow for full training.

### 1. Clone the repository

```bash
git clone git@github.sydney.edu.au:SOFT3888-TU14-01/SOFT3888_TU14_01.git
cd SOFT3888_TU14_01
```

### 2. Create the environment

**Linux or Windows with an NVIDIA GPU (conda/mamba):**

```bash
mamba env create -f environment.yml
mamba activate dsec-env
```

`conda env create -f environment.yml` and `conda activate dsec-env` work the same way.

**macOS, or any machine without an NVIDIA GPU (venv + pip):**

`environment.yml` pins `pytorch-cuda=12.1`, which does not exist for macOS, so `mamba env create` fails there with `pytorch-cuda =12.1 * does not exist`. Use a virtual environment instead:

```bash
python3.11 -m venv .venv           # Windows: py -3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision "ultralytics==8.4.128" "hdf5plugin==7.0.0" "kornia==0.8.3" \
  h5py numba numpy opencv-python pillow pyyaml seaborn tqdm
```

Keep the environment active for every step that follows.

### 3. Install the DSEC detection utilities

The repo depends on the external `DSEC-det` package for dataset parsing and labelling support. Clone it **inside the project root**: `dataset_loader.py` reads the train/val/test split from `DSEC-det/config/train_val_test_split.yaml` relative to the project root, and stops with `FileNotFoundError` if the folder is anywhere else.

```bash
git clone https://github.com/uzh-rpg/DSEC-det.git
cd DSEC-det
pip install -e .
cd ..
```

`DSEC-det/` is listed in `.gitignore`, so it never shows up in `git status`.

### 4. Pretrained weights

Training starts from Ultralytics' pretrained `yolo26n.pt`. It is not committed (`*.pt` is in `.gitignore`); Ultralytics downloads it into the project root on the first training run. On a machine without internet access, download `yolo26n.pt` from the [Ultralytics assets releases](https://github.com/ultralytics/assets/releases) and place it in the project root.

### 5. Check the installation

```bash
python -c "import torch, ultralytics, cv2, h5py, hdf5plugin, kornia, dsec_det; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| CUDA', torch.cuda.is_available())"
python -m unittest discover -s tests
```

The first command should print the versions without an `ImportError`. The test suite does not need the dataset. One test is skipped unless the `DSEC_ROOT` environment variable points to the dataset.

## Data layout

The code uses the [DSEC-Detection](https://dsec.ifi.uzh.ch/dsec-detection/) dataset: the original DSEC sequences plus the DSEC-Detection extension (object labels and six extra sequences). The dataset is not in this repository and can live anywhere on disk; pass its root folder to every script with `--dataset-path`.

### Downloading the dataset

The `DSEC-det` clone from step 3 contains the official download scripts. Run all three from inside `DSEC-det/`, with an **absolute** path for `DSEC_ROOT`: the scripts `cd` into sub-folders, so a relative path puts files in the wrong place.

```bash
cd DSEC-det
DSEC_ROOT=/absolute/path/to/DSEC_ROOT
bash scripts/download_dsec.sh "$DSEC_ROOT"             # original DSEC: events, images, calibration
bash scripts/download_dsec_extra.sh "$DSEC_ROOT"       # DSEC-Detection: new sequences and object labels
bash scripts/download_remapped_images.sh "$DSEC_ROOT"  # images remapped into the event camera view
cd ..
```

`download_dsec.sh` also fetches disparity and optical flow, which this project does not use. The download is large: a 25-sequence partial copy takes 59 GB, so allow roughly 150 GB for all 60 sequences, plus temporary space for the zip files. See the [DSEC-det README](https://github.com/uzh-rpg/DSEC-det) for details.

### Expected structure

```text
DSEC_ROOT/
├── train/                          training AND validation sequences
│   ├── thun_00_a/
│   │   ├── calibration/
│   │   ├── events/left/events.h5
│   │   ├── images/timestamps.txt
│   │   ├── images/left/distorted/*.png
│   │   └── object_detections/left/tracks.npy
│   ├── ...
│   └── zurich_city_21_a/
└── test/                           test sequences
    ├── interlaken_00_a/
    ├── ...
    └── zurich_city_15_a/
```

The split is set by `DSEC-det/config/train_val_test_split.yaml`:

| Split | Sequences | Read from |
| --- | ---: | --- |
| train | 41 | `DSEC_ROOT/train/` |
| val | 6 (`zurich_city_16_a` to `zurich_city_21_a`) | `DSEC_ROOT/train/` |
| test | 13 | `DSEC_ROOT/test/` |

Each sequence needs all four of: `images/timestamps.txt`, the distorted RGB frames in `images/left/distorted/`, the events in `events/left/events.h5` (read by the teacher and distillation runs), and the labels in `object_detections/left/tracks.npy`.

### Incomplete dataset downloads

**The code may not run correctly unless the dataset is downloaded in its entirety.** A partial or interrupted download does not always stop with a clear error. Depending on what is missing, you get one of the following:

| What is missing | What happens |
| --- | --- |
| Some sequence folders | **No error.** `DSECDet` silently skips any sequence in the split file that is not on disk, so training runs on fewer sequences and the results will not match the report. Compare the `Training Sequences` and `Validation Sequences` lists in the run's `train.log` or `config.yaml` with the table above. |
| Some of a sequence's distorted images | **Usually no error.** Frames are matched to labels by their position in the sorted image list, so every frame after a gap is paired with the wrong labels. If enough images are missing, the sequence's last frames raise `IndexError`. |
| `DSEC_ROOT/test/` or all test sequences | Training runs, then stops at the final test step with `ValueError: The test split is empty; provide a labelled test dataset.` `evaluate.py` stops with the same error. |
| All training sequences | Stops before the first epoch with `ValueError: num_samples should be a positive integer value, but got num_samples=0`. |
| All validation sequences | Stops before the first epoch with `ValueError: Training and validation loaders must both be non-empty.` |
| A sequence's `images/left/distorted/` folder | `IndexError: list index out of range` at the first batch that reaches that sequence. |
| `timestamps.txt`, `tracks.npy` or `events.h5` | `FileNotFoundError` while the dataset loads or at the first affected batch. RGB-only runs never open `events.h5`, so a missing event file only breaks teacher and distillation runs. |
| Part of a file, from an interrupted download or unzip | An `OSError` from h5py or NumPy when the file is opened or read, or an OpenCV error on an unreadable image. |
| Any of the 41 training sequences, for `--distill-mode object-quality` | `tools/measure_event_density.py` refuses to run and lists the missing sequences, so `docs/event_density_reference.json` is never created and `object-quality` training cannot start. |

If any of these appear, re-run the download scripts for the affected split and confirm each sequence folder has the four items listed above. To check the code on a smaller copy without a full download, use `--max-samples` (see [Smoke checks](#smoke-checks-and-diagnostics)); results from a partial dataset are not comparable with the report.

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
- shuffles the training data with a seeded generator (`--seed`, default 0),
- augments each training frame (flip, scale, shift, hue/saturation) unless `--no-augment` is given,
- collates labels into a YOLO-compatible target structure,
- accumulates gradients to about 64 images per step and clips them to avoid exploding updates,
- validates after each epoch,
- saves the model with the highest validation mAP50-95,
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
draw a strength. Darkening uses `darkening.darken`, mapped onto this strength
scale exactly as in the benchmark (see below).
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

The best checkpoint is selected using the highest validation mAP50-95. Validation loss is not used, because the YOLO26 loss shifts its weighting between the two detection heads every epoch, so loss values are not comparable across epochs.

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

Darkening is applied by `darkening.darken`, which works on an illumination scale instead (1 = unchanged). The benchmark strength maps linearly onto it: `illumination = 1 - 0.8 * strength`, so strength 0 leaves images unchanged, `mild` (0.35) dims to 72% illumination and `strong` (1.0) dims to darkening's own `strong` level (20%). Each frame's illumination field and sensor noise come from its per-frame seed. The degradation-augmentation baseline uses the same mapping, so its training darkening matches this benchmark condition.

The evaluation seed keeps benchmark inputs reproducible. Each condition also produces an RGB fingerprint, which can be compared between runs to confirm that the same input images were evaluated.

For fair comparisons between models, use the same dataset split, degradation settings, evaluation seed, and benchmark configuration.

### Export and compare benchmark results

After the evaluation code is committed, run each checkpoint with the same dataset release, seed, strength, and sample limit. Add these options to the benchmark command above:

```bash
  --model-name rgb_baseline \
  --dataset-id dsec-left-annotations-v1 \
  --benchmark-report runs/benchmark_reports/rgb_baseline.json
```

Choose a `--dataset-id` that identifies the **actual image and annotation version** used by the team; the example is only a placeholder. Use a different report path and one of `rgb_baseline`, `rgb_augmented`, `naive_distillation`, `proposed_event`, or `teacher` for each checkpoint. For the teacher, use `--type teacher --model-name teacher`. The exported JSON records the checkpoint path and SHA-256 hash, code commit and clean/dirty state, dataset ID and path, test split, seed, strength, sample limit, parameter count, metrics, and per-condition RGB fingerprints. It contains measured results only; it does not create a score for a model that has not been evaluated.

When reports are available, combine them with:

```bash
python tools/compare_benchmark_reports.py \
  runs/benchmark_reports/rgb_baseline.json \
  runs/benchmark_reports/rgb_augmented.json \
  --output-dir runs/benchmark_comparison
```

This writes `comparison.json` and `comparison.csv`. Output files must be new: rerunning with an existing report path or comparison output directory containing these files raises an error instead of replacing them. Missing milestone models appear as `missing` with blank metrics. The comparison rejects duplicate model names or checkpoints, uncommitted or unknown evaluation code, mismatched code commits, dataset IDs, test splits, strengths, seeds, sample limits, or RGB input fingerprints. A matching dataset ID is supplied by the team and cannot itself verify that annotations are identical; use the same agreed dataset copy and annotation version. Training checkpoints may come from different commits, but evaluate all of them with one committed evaluation-code version. Batch size and worker count are recorded but may differ if the input fingerprints still match. Runtime measurements and qualitative examples require separate collection before the main evaluation milestone is complete.

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
- The training pipeline saves both a best checkpoint and a latest checkpoint after every epoch. To continue an interrupted run in the same folder, pass only its folder: `python train_rgb_student.py --resume runs/student/exp_006`. The settings come from the original run (`run_args.yaml`, or `run_command.txt` for older runs); only `--num-workers` and `--console-level` may be given again. Model, optimiser, EMA and schedules continue exactly, but the shuffle order and augmentation draws after the resume point differ from an uninterrupted run.

## Testing and validation

Run the unit and integration checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

The test suite covers the project’s expected data-loading and training workflow behaviour. It uses synthetic data, so it runs without the dataset. `tests/test_overexposure_night.py` also checks real DSEC night frames when the `DSEC_ROOT` environment variable is set (`DSEC_ROOT=/path/to/DSEC_ROOT python -m unittest discover -s tests`); otherwise it is skipped.

## Smoke checks and diagnostics

The quickest end-to-end check is a one-epoch run on a few frames. It takes seconds, even on a CPU:

```bash
python train_rgb_student.py \
  --dataset-path "/path/to/DSEC_ROOT" \
  --epochs 1 \
  --max-samples 8 \
  --skip-test
```

It should finish with `Test evaluation skipped (--skip-test).` and write `best_student_yolo.pt` to a new `runs/student/exp_NNN/`. Drop `--skip-test` to also exercise the test split. Swap in `train_teacher.py` to check the event files as well.

The repository also includes a helper for a fuller integration check on small disjoint subsets. It needs both `train/` and `test/` under the dataset root and a new output folder:

```bash
python tools/run_issue17_smoke.py --dataset-root "/path/to/DSEC_ROOT" --output-dir "runs/issue17_verify"
```

On Windows with a venv: `.\.venv\Scripts\python.exe tools/run_issue17_smoke.py --dataset-root "D:\DSEC_ROOT" --output-dir "D:\DSEC_issue17_verify"`.

These checks validate the pipeline without a full training run across the whole dataset.

## Expected outputs

Each training run creates a new numbered folder, `runs/<role>/exp_NNN/`, where `<role>` is `student`, `teacher` or `distill`. It contains:

- `train.log`: the configuration, per-batch progress and per-epoch metrics
- `config.yaml`, `run_args.yaml` and `run_command.txt`: the settings used, which `--resume` reads
- `results.csv` and `results.png`: per-epoch metrics and training curves
- `best_<role>_yolo.pt`: the checkpoint with the highest validation mAP50-95
- `latest_<role>_yolo.pt`: the most recent checkpoint, for resuming
- `test/`: confusion matrix, PR curves and example predictions from the final test step (absent with `--skip-test`)

## Troubleshooting

If training or evaluation fails, check the following first:

1. Ensure the dataset path points to the DSEC root, the folder that contains `train/` and `test/`.
2. Confirm the dataset is complete (see [Incomplete dataset downloads](#incomplete-dataset-downloads)).
3. `ModuleNotFoundError: No module named 'dsec_det'`: install `DSEC-det` in the active environment (`pip install -e DSEC-det`).
4. `FileNotFoundError` for `DSEC-det/config/train_val_test_split.yaml`: `DSEC-det` must be cloned inside the project root.
5. `mamba env create` fails with `pytorch-cuda =12.1 * does not exist`: you are on macOS; use the venv route in step 2.
6. Run every script from the project root, so `yolo26n.pt`, `docs/` and `runs/` are found.
7. Ensure the checkpoint path passed to `--weights` or `--teacher-checkpoint` exists.
8. Check that the environment matches the required Python and dependency versions.

This project is meant to serve as a full training and evaluation workflow for DSEC-based RGB and RGB + event detection experiments. It is especially useful for experimenting with student–teacher knowledge transfer and multi-modal perception pipelines in a structured and reproducible setup.