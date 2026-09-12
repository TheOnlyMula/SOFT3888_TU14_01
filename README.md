# SOFT3888_TU14_01

This project will develop an event-privileged training framework using the DSEC-Detection dataset. A teacher model will use paired RGB and event-camera information to perform traffic object detection, while an RGB-only student model will learn from the teacher through knowledge distillation. This approach will investigate whether object-aware distillation can improve the transfer of information from the teacher to the student. The resulting student model will operate independently using RGB input only during inference. 

Our capstone project for SOFT3888 aims to address the drawbacks of RGB cameras in object detection by integrating data from event cameras into model training. If successful, the primary benefit of this approach is detecting objects in traffic more accurately with reduced computational overhead in a more lightweight framework. This allows for faster response times while maintaining accuracy in adequate lighting conditions, and surpassing the accuracy of conventional RGB systems in harsh lighting conditions. 

## Repository structure

```text
SOFT3888_TU14_01/
├── README.md
├── dataset_loader.py
├── train_rgb_student.py
├── train_teacher.py
├── evaluate.py
├── utils.py
├── log.py
├── environment.yml
├── yolo26n.pt
├── configs/
│   ├── dsec_rgb.yaml
│   ├── yolo26n_student.yaml
│   └── yolo26n_teacher.yaml
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

The dataset uses the DSEC detector utilities and a synchronization mode of `sync="back"`, which means each RGB image is paired with the event stream that occurred immediately before the image acquisition, preserving causal ordering.

### train_rgb_student.py

This script trains the student RGB-only YOLO model. It:

- builds a DSEC RGB dataloader for train/validation splits,
- initialises a YOLO model from the pretrained `yolo26n.pt` checkpoint,
- trains with a YOLO-style loss,
- logs training metrics,
- saves the best validation checkpoint,
- optionally runs a final test evaluation.

### train_teacher.py

This script trains the teacher model using RGB + event fusion. It follows the same training loop pattern as the student version but loads the fused dataloader, which includes both RGB image channels and the event histogram.

### evaluate.py

This file includes the validation and test evaluation logic used by both training scripts. It:

- validates a checkpoint on a validation loader,
- computes batch-wise loss summaries,
- restores a saved checkpoint into the Ultralytics model,
- evaluates on the DSEC test split,
- logs loss and metrics for analysis.

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

### Teacher RGB + event training

The teacher training uses the fused dataloader and the same overall training pattern, but adds event information to the input tensor.

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

The `--type` argument determines which model config is used:

- `student` -> `configs/yolo26n_student.yaml`
- `teacher` -> `configs/yolo26n_teacher.yaml`

The evaluation loads the saved PyTorch checkpoint state, restores it into the Ultralytics model, and validates it on the test split.

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
