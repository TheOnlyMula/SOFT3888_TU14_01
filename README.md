# SOFT3888_TU14_01

This project will develop an event-privileged training framework using the DSEC-Detection dataset. A teacher model will use paired RGB and event-camera information to perform traffic object detection, while an RGB-only student model will learn from the teacher through knowledge distillation. This approach will investigate whether object-aware distillation can improve the transfer of information from the teacher to the student. The resulting student model will operate independently using RGB input only during inference.

Our capstone project for SOFT3888 aims to address the drawbacks of RGB cameras in object detection by integrating data from event cameras into model training. If successful, the primary benefit of this approach is detecting objects in traffic more accurately with reduced computational overhead in a more lightweight framework. This allows for faster response times while maintaining accuracy in adequate lighting conditions, and surpassing the accuracy of conventional RGB systems in harsh lighting conditions.

## Repository structure

```text
SOFT3888_TU14_01/
├── train_rgb_student.py        entry points: thin scripts that call into the packages
├── train_teacher.py
├── train_distilled_student.py
├── evaluate.py
├── overfit.py
├── loaders/                    DSEC dataset, causal event windows, DataLoaders, augmentation
│   ├── dataset_loader.py
│   └── augment.py
├── degradation/                RGB degradations for the benchmark and degradation augmentation
│   ├── strength.py
│   ├── conditions.py
│   ├── darkening.py
│   └── overexposure.py
├── training/                   training loop, losses and the training-only distillation code
│   ├── cli.py, run.py, loop.py, losses.py, schedule.py, reporting.py
│   ├── distillation.py
│   └── overfit.py
├── evaluation/                 validation, test and benchmark metrics; evaluate.py's command line
│   ├── metrics.py
│   ├── benchmark_reporting.py
│   └── cli.py
├── utils/                      shared helpers
│   ├── model.py
│   └── log.py
├── configs/                    one frozen config per experiment in the main comparison
├── docs/                       degradation guides and measured constants (event_scale.json, event_density.json)
├── tests/                      regression and smoke tests
├── tools/                      measurement, example-generation, comparison and visualisation scripts
├── environment.yml
├── yolo26n.pt
├── DSEC_ROOT/                  labelled DSEC sequences (not in git)
├── DSEC-det/                   official DSEC detection utilities (not in git)
└── runs/                       training outputs and logs (not in git)
```

## Core components

### loaders/dataset_loader.py

This file defines the DSEC dataset access layer. It creates a custom dataset that:

- reads RGB images from DSEC sequences,
- optionally reads event tensors from event files,
- converts event polarity information into a 2-channel histogram,
- maps labels into YOLO-style bounding box targets,
- supports both RGB-only and fused RGB + event training paths,
- augments the training split when `augment=True` (see below).

Images and labels come from the DSEC detector utilities. For real-time detection, each RGB image is paired only with events from the 50 ms _before_ its timestamp (`load_causal_events` in `loaders/dataset_loader.py`), so the model never sees events that happen after the image. DSECDet's own `sync` windows are not used for events: `sync="back"` returns the 50 ms after the image, and `sync="front"` pairs the image with the previous frame's labels. The first frame of each sequence has no preceding events, so its event channels are empty.

### loaders/augment.py

Training-time augmentation, used by every model so student comparisons stay matched. One random draw moves the RGB pixels, the event pixels and the boxes together, so events stay aligned with the RGB the student sees:

- horizontal flip (probability 0.5),
- zoom of 0.5-1.5x about the centre and a shift of up to 10% of the width and height,
- hue shift and saturation jitter on the RGB channels only (hue moves by the same amount for every colour, as in Ultralytics 8.4). Brightness is left alone because brightness shifts belong to the degradation-augmentation baseline.

Pixels uncovered by the zoom or shift are grey (114) in RGB and 0 in the event channels, since any other event value would invent events. Boxes that end up mostly outside the image, or cut into slivers more than 100 times longer than wide, are dropped. Every sample makes the same random draws with or without events, so for the same `--seed` the RGB student and the distilled student see identical augmented RGB.

### degradation/

The RGB degradations. `darkening.py` and `overexposure.py` are the image transforms (see the end of this README); both take the shared strength scale of `strength.py`, where 0 leaves an image unchanged and `mild` (0.35) and `strong` (1.0) name the benchmark levels. `conditions.py` seeds them deterministically for the degraded evaluation benchmark and samples the random conditions of the degradation-augmentation training baseline (`--degradation-augmentation`).

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

Trains the RGB student with feature distillation from a frozen RGB + event teacher (`training/distillation.py`). Training batches carry RGB + event channels: the teacher sees all of them, the student only the RGB ones. Validation and test stay RGB-only, so student results never depend on event data.

```bash
python train_distilled_student.py --config configs/object_distillation.yaml --dataset-path dataset --num-workers 8

# Same run with each image's object-aware loss scaled by the quality weight w
python train_distilled_student.py --config configs/proposed_event.yaml --dataset-path dataset --num-workers 8
```

The distillation loss is the smooth L1 distance between the student's and the frozen teacher's one-to-many P3 neck features (stride 8). Student and teacher share an architecture, so the features match without a projection layer. `--distill-mode` chooses where it is applied:

- `unweighted` (default, the baseline): every spatial location counts equally.
- `object`: only the labelled objects count (proposal eq. 2). Each image's boxes, after augmentation, are enlarged by `MASK_ENLARGEMENT` (1.2) about their centres and marked on the feature grid; the distance is averaged over those cells. Images without objects add no distillation loss. With a mask covering everything, this equals the unweighted loss, so one `--lambda-distill` serves both modes.
- `object-quality`: `object` with each image's loss multiplied by the deterministic quality weight `w = clip(s_e * c_t, 0, 1)` (proposal eq. 3), so unreliable event windows and unsure teacher outputs supervise the student less. `w` is the only difference between the two modes: with every `w = 1` they train identically.

#### Quality weight (`object-quality` mode)

For each training image `i`, `training.distillation.quality_weight` combines:

- `s_e`, the normalised event density. The dataset measures the coverage of the image's causal 50 ms event window (`loaders.dataset_loader.event_coverage`: the fraction of pixels where either polarity saw at least one event) before augmentation, which would otherwise change it by zooming and interpolating the histogram. `training.distillation.event_density` divides it by the fixed `EVENT_DENSITY_REFERENCE` and clips it to 1, so no events gives 0 and a window at least as dense as the reference gives 1.
- `c_t`, the mean confidence of the frozen teacher's detections at or above `TEACHER_CONFIDENCE_THRESHOLD` (0.25), or 0 when there are none (`training.distillation.teacher_confidence`). The teacher is loaded end-to-end, so its output is already the final detections and no NMS is needed.
- `w_i = clip(s_e * c_t, 0, 1)`. Neither factor carries a gradient, so nothing in `w` is learned.

The batch distillation loss is

```
L_distill = (1 / B) * sum_i  w_i * L_distill_i
```

where `L_distill_i` is image `i`'s object-aware loss and `B` is the number of images in the batch. The mean runs over every image, including those with `w_i = 0`, so down-weighting an image lowers the batch loss rather than handing its share to the other images. With every `w_i = 1` this is exactly the `object` loss. It enters the total loss as `lambda_distill * L_distill` (proposal eq. 1), as in the other modes.

`EVENT_DENSITY_REFERENCE` is the median event coverage over every training frame, so a window at least as dense as the typical frame counts as fully reliable and only sparser windows are down-weighted. It is measured once with `python tools/measure_event_density.py --dataset-path /path/to/DSEC` and recorded in `docs/event_density.json`, in the same way `EVENT_SCALE_K` is recorded in `docs/event_scale.json`.

Each epoch adds the mean, minimum and maximum `w` and the fraction of images with `w = 0` to `train.log`:

```
  train quality_weight mean = 0.4120 min = 0.0000 max = 0.9310 zero_fraction = 0.0850
```

and to `results.csv` as `train/quality_weight_mean`, `train/quality_weight_min`, `train/quality_weight_max` and `train/quality_weight_zero_fraction` (not drawn in `results.png`). Every training image's values are also appended to `quality_weights.csv` in the run folder, one row per image and epoch (`epoch, sequence, frame, event_density, teacher_confidence, quality_weight`), for visualising the weights and finding the frames they down-weight.

The mask and the quality weight are built in the training step only and are never part of evaluation or inference. Each epoch logs `distill_loss`, the mode and the weighted contribution to the total loss; `config.yaml` records the distillation mode, the teacher confidence threshold, the mask enlargement (`object` and `object-quality`) and the event-density reference (`object-quality`).

### train_rgb_student.py and train_teacher.py

Thin entry points into the `training` package. The two models are trained identically; the only difference is the input:

- `train_rgb_student.py`: 3 channels (RGB).
- `train_teacher.py`: 5 channels (RGB + 2-channel event histogram).

Both accept the same arguments (`--dataset-path`, `--epochs`, `--batch-size`, `--num-workers`, `--max-samples`, `--skip-test`, `--seed`, `--no-augment`, `--resume`, `--console-level`). Use the same `--seed` for every student you compare; it fixes model initialisation, the shuffle order and the augmentation draws.

### evaluation/ and evaluate.py

`evaluation/metrics.py` holds the evaluation used for both per-epoch validation and the final test step, so validation and test numbers are computed identically. `evaluate.py` is the command-line entry point (its options are in `evaluation/cli.py`), and `evaluation/benchmark_reporting.py` exports and compares benchmark results. The evaluation package never imports `training`, so the RGB-only evaluation path stays free of the teacher and the other training-only components. In a single pass over the data it:

- computes the per-image loss (box, class, L1),
- computes precision, recall, mAP@0.5 and mAP@0.5:0.95,
- reports per-class results,
- for the test step, restores the best checkpoint and saves the confusion matrix, PR curves and example predictions to `runs/<role>/exp_NNN/test/`.

### utils/

`utils/log.py` creates the numbered run folders, loggers, `config.yaml` and `results.csv`. `utils/model.py` contains shared functions for:

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

### Experiment configs

Each model in the main comparison (proposal section 4.1) has a frozen config in `configs/`, so every run of it uses the same settings and the five students are guaranteed to match in epochs, batch size, seed and augmentation (a test checks this). Only `rgb_augmented` trains on degraded images: the teacher and the distilled students train on undegraded images, so the distilled students learn any robustness from the teacher's event supervision rather than from seeing degradations, and `rgb_augmented` is the alternative they are compared against. The teacher alone uses modality dropout (`rgb_dropout: 0.15`, see the teacher section). Pass it with `--config`, and give the machine-specific options on the command line:

```bash
python train_rgb_student.py --config configs/rgb_baseline.yaml --dataset-path dataset --num-workers 8
```

| Config | Script | Comparison row | Benchmark `--model-name` |
| --- | --- | --- | --- |
| `rgb_baseline.yaml` | `train_rgb_student.py` | 1. RGB-only student | `rgb_baseline` |
| `rgb_augmented.yaml` | `train_rgb_student.py` | 2. RGB student with degradation augmentation | `rgb_augmented` |
| `naive_distillation.yaml` | `train_distilled_student.py` | 3. Basic unweighted distillation | `naive_distillation` |
| `object_distillation.yaml` | `train_distilled_student.py` | 4. Object-aware distillation | `object_distillation` |
| `proposed_event.yaml` | `train_distilled_student.py` | 5. Quality-weighted object-aware distillation | `proposed_event` |
| `teacher.yaml` | `train_teacher.py` | 6. Privileged teacher (training reference) | `teacher` |

A config's keys are the command-line options with underscores (`batch_size: 6` is `--batch-size 6`, `degradation_augmentation: true` is the bare flag). Options typed on the command line override the config, e.g. `--epochs 1 --max-samples 16` for a quick check. The distillation configs read the frozen teacher from `checkpoints/teacher.pt`: once the teacher is trained, copy its `best_teacher_yolo.pt` there (`.pt` files are not committed). Each run records the full command with the config's settings spelled out (`run_command.txt`, `run_args.yaml`) and the config's path (`config.yaml`), so `--resume` repeats the run exactly even if the config is edited later; `--config` cannot be combined with `--resume`.

All the options below can also be set in a config.

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

Enable a random degradation mix for training:

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
draw a strength. Darkening uses `degradation.darkening.darken`, mapped onto this strength
scale exactly as in the benchmark (see below).
With event channels (the teacher and the distilled students) only the RGB
channels are degraded and the events stay clean, so the teacher learns to rely on
them when the RGB is poor. These options do not change benchmark settings.

`--augmentation-seed` (default `0`, non-negative integer) controls a dedicated
random stream for condition selection, strength and darkening noise. It is
independent of the benchmark's `--evaluation-seed` and global NumPy/PyTorch RNGs.
Each loader worker has its own stream, which advances across epochs. The same
seed reproduces the augmentation sequence for the same sample order, batch size
and worker count; this seed does not control training shuffle or model randomness.
Changing the worker count can change the augmentation sequence, so give every run in the
comparison the same `--num-workers`: then the distilled students see exactly the same degraded
RGB views as `rgb_augmented`. Augmentation RNG
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

This path uses the event histogram generated from event polarity, stacking it alongside RGB features before feeding the model. The teacher trains with the ordinary detection loss only: the object mask and the quality weight belong to the student's distillation loss. Its first layer takes 5 channels, so it cannot load the pretrained 3-channel weights directly; instead its RGB filters are copied from them and its event filters start at zero (`build_model` in `utils/model.py`). The teacher therefore starts as exactly the pretrained RGB detector, like the student, and learns from there how much to use the events. A teacher trained this way leans almost entirely on RGB: its RGB path starts pretrained, the DSEC labels come from an RGB tracker, and most frames are well lit, so zeroing its events cost only 0.013 mAP50-95. `configs/teacher.yaml` therefore uses modality dropout (`--rgb-dropout 0.15`): the RGB of 15% of training images is blanked while their events and labels are kept, so the teacher must also learn to detect from events. The rate follows Tomy et al. (ICRA 2022), who trained RGB + event detectors on DSEC with image dropout at 0.15. It applies to the teacher only; a student's RGB is never blanked.

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

Both models use the stock Ultralytics YOLO26n architecture with an 8-class DSEC head (built by `build_model` in `utils/model.py`). The `--type` argument sets the number of input channels:

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

`--degradation-strength` accepts the named levels `mild` (0.35) and `strong` (1.0) defined in `degradation/strength.py`, or a number in [0, 1]. The same value is used for the darkened (`degradation.darkening.darken`) and over-exposed (`degradation.overexposure.overexpose`) conditions.

Darkening is applied by `degradation.darkening.darken`, which works on an illumination scale instead (1 = unchanged). The benchmark strength maps linearly onto it: `illumination = 1 - 0.8 * strength`, so strength 0 leaves images unchanged, `mild` (0.35) dims to 72% illumination and `strong` (1.0) dims to darkening's own `strong` level (20%). Each frame's illumination field and sensor noise come from its per-frame seed. The degradation-augmentation baseline uses the same mapping, so its training darkening matches this benchmark condition.

The evaluation seed keeps benchmark inputs reproducible. Each condition also produces an RGB fingerprint, which can be compared between runs to confirm that the same input images were evaluated.

For fair comparisons between models, use the same dataset split, degradation settings, evaluation seed, and benchmark configuration.

### Export and compare benchmark results

After the evaluation code is committed, run each checkpoint with the same dataset release, seed, strength, and sample limit. Add these options to the benchmark command above:

```bash
  --model-name rgb_baseline \
  --dataset-id dsec-left-annotations-v1 \
  --benchmark-report runs/benchmark_reports/rgb_baseline.json
```

Choose a `--dataset-id` that identifies the **actual image and annotation version** used by the team; the example is only a placeholder. Use a different report path and one of `rgb_baseline`, `rgb_augmented`, `naive_distillation`, `object_distillation`, `proposed_event`, or `teacher` for each checkpoint (the names of the configs in `configs/`). For the teacher, use `--type teacher --model-name teacher`. The exported JSON records the checkpoint path and SHA-256 hash, code commit and clean/dirty state, dataset ID and path, test split, seed, strength, sample limit, parameter count, metrics, and per-condition RGB fingerprints. It contains measured results only; it does not create a score for a model that has not been evaluated.

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
- The training pipeline saves both a best checkpoint and a latest checkpoint after every epoch. To continue an interrupted run in the same folder, pass only its folder: `python train_rgb_student.py --resume runs/student/exp_006`. The settings come from the original run (`run_args.yaml`); only `--num-workers` and `--console-level` may be given again. Model, optimiser, EMA and schedules continue exactly, but the shuffle order and augmentation draws after the resume point differ from an uninterrupted run.

## Testing and validation

Run the unit and integration checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

The test suite covers the project’s expected data-loading and training workflow behaviour.

## Smoke checks and diagnostics

For a quick end-to-end check without a full run, limit every split to a few evenly spaced frames:

```bash
python train_rgb_student.py --dataset-path dataset --epochs 1 --max-samples 16
```

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

## Deterministic overexposure

`overexpose(image, strength)` returns a new, same-shape `uint8` RGB array
without randomness or input mutation. Input is a non-empty `H x W x 3` array;
convert BGR to RGB and apply it before tensor normalization or event concatenation.

```python
from degradation.overexposure import overexpose
mild_rgb = overexpose(rgb, "mild")
strong_rgb = overexpose(rgb, "strong")
custom_rgb = overexpose(rgb, 0.6)  # Continuous strength in [0, 1].
```

| Preset | Strength | Gamma | Normalized offset |
| --- | ---: | ---: | ---: |
| mild | 0.35 | 0.79 | 0.14 |
| strong | 1.00 | 0.40 | 0.40 |

The offset was calibrated on 400 training frames so that `strong` saturates (all RGB channels equal 255) a median 23% of a typical frame: severe, but objects stay visible. An earlier offset of 0.6 whitened a median 72.5% of daytime frames, and every model scored 0 mAP on it; see the [detailed guide](docs/overexposure/README.md#calibration).

[Day examples](docs/overexposure/day_000007.png) and
[night examples](docs/overexposure/night_000300.png) show clean/mild/strong.
See the [detailed guide](docs/overexposure/README.md) for the formula and measured results.

Generate examples and descriptive measurements into a new output directory:

```bash
python tools/generate_overexposure_examples.py --image "/path/to/rgb.png" --output-dir "runs/overexposure_examples/sample_01"
```

## Deterministic darkening

`darken(image, strength, seed=0)` takes:
- `image`: a non-empty `H x W x 3` RGB array with dtype `uint8` (convert OpenCV's BGR to RGB first),
- `strength`: the benchmark's shared strength, `"mild"` (0.35), `"strong"` (1.0) or a number in [0, 1], exactly as for `overexpose`. 0 leaves the image unchanged,
- `seed`: an integer fixing the illumination draw, the illumination field and the noise.

It returns a dictionary:
```python
{
  'image': ...,                # a new, same-shape uint8 darkened RGB array with sensor noise
  'seed': ...,                 # the seed used
  'global_illum_factor': ...,  # the global illumination factor drawn for this image (see below)
  'mean_brightness': ...,      # mean linear brightness before noise
  'proportion_dark': ...,      # proportion of pixels at or below 1% linear light in every channel, before noise
}
```

```python
from degradation.darkening import darken
mild_rgb = darken(rgb, "mild", seed=1234)["image"]
strong_rgb = darken(rgb, "strong", seed=1234)["image"]
custom_rgb = darken(rgb, 0.6, seed=1234)["image"]  # any strength in [0, 1]
```

The illumination level falls linearly with strength, from 1 (unchanged) at strength 0 to `STRONG_ILLUMINATION` (0.2) at strength 1, and each image's global illumination factor is drawn uniformly within ±0.1 of that level, so no two frames are darkened identically:

| Strength | Illumination level | Global illumination factor range |
| --- | ---: | ---: |
| 0 | 1.00 | unchanged |
| mild (0.35) | 0.72 | [0.62, 0.82) |
| strong (1.0) | 0.20 | [0.10, 0.30) |

The image is dimmed in linear light by that factor times a smooth random illumination field (0.7 to 1.3), then given Poisson photon shot noise and Gaussian read noise.

[Day examples](docs/darkening/day_000003.png) and
[night examples](docs/darkening/night_000005.png) show the clean, mild and strong results, with the illumination factor, mean brightness and percentage of dark pixels, for `interlaken_00_c/000003.png` and `zurich_city_09_a/000005.png` (seed 1234).
See the [detailed guide](docs/darkening/README.md) for more information.

Generate examples and descriptive measurements into a new output directory:

```bash
python tools/generate_darkened_examples.py --image "/path/to/rgb.png" --output-dir "runs/darkening_examples/sample_01"
```
