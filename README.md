# SOFT3888_TU14_01

This project will develop an event-privileged training framework using the DSEC-Detection dataset. A teacher model will use paired RGB and event-camera information to perform traffic object detection, while an RGB-only student model will learn from the teacher through knowledge distillation. This approach will investigate whether object-aware distillation can improve the transfer of information from the teacher to the student. The resulting student model will operate independently using RGB input only during inference.

Our capstone project for SOFT3888 aims to address the drawbacks of RGB cameras in object detection by integrating data from event cameras into model training. If successful, the primary benefit of this approach is detecting objects in traffic more accurately with reduced computational overhead in a more lightweight framework. This allows for faster response times while maintaining accuracy in adequate lighting conditions, and surpassing the accuracy of conventional RGB systems in harsh lighting conditions.

Event-privileged training on DSEC-Detection: an RGB + event teacher is trained first, then RGB-only students learn from it through knowledge distillation. This README covers setting up the environment, preparing for training, and running training.

Run every command from the project root unless a step says otherwise.

## 1. Environment setup

### Prerequisites

- Git, and either mamba/conda or Python 3.11.
- An NVIDIA GPU with CUDA for full training runs. Without one the code falls back to the CPU, which is enough for the tests and a smoke run but too slow for full training.

### Clone the repository

```bash
git clone git@github.sydney.edu.au:SOFT3888-TU14-01/SOFT3888_TU14_01.git
cd SOFT3888_TU14_01
```

### Create the environment

**Linux or Windows with an NVIDIA GPU:**

```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
mamba activate dsec-env               # or: conda activate dsec-env
```

**macOS, or any machine without an NVIDIA GPU:** `environment.yml` pins `pytorch-cuda=12.1`, which does not exist on macOS, so use a virtual environment instead:

```bash
python3.11 -m venv .venv           # Windows: py -3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision "ultralytics==8.4.128" "hdf5plugin==7.0.0" "kornia==0.8.3" \
  h5py numba numpy opencv-python pillow pyyaml seaborn tqdm
```

Keep the environment active for every step that follows.

### Install the DSEC detection utilities

Clone `DSEC-det` inside the project root. The dataset loader reads the train/val/test split from `DSEC-det/config/train_val_test_split.yaml` relative to the project root, and fails with `FileNotFoundError` if the folder is anywhere else.

```bash
git clone https://github.com/uzh-rpg/DSEC-det.git
pip install -e DSEC-det
```

`DSEC-det/` is in `.gitignore`.

### Check the installation

```bash
python -c "import torch, ultralytics, cv2, h5py, hdf5plugin, kornia, dsec_det; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
python -m unittest discover -s tests
```

The first command should print the versions without an `ImportError`. The tests use synthetic data and do not need the dataset; one test is skipped unless `DSEC_ROOT` points to it.

## 2. Preconfiguration

### Pretrained weights

Training starts from Ultralytics' `yolo26n.pt`, which is downloaded into the project root on the first run. Offline, download it from the [Ultralytics assets releases](https://github.com/ultralytics/assets/releases) and put it in the project root.

### Experiment configs

Each model in the comparison has a frozen config in `configs/`, so all five students match in epochs (25), batch size (6) and seed (0):

| Config | Script | Model |
| --- | --- | --- |
| `teacher.yaml` | `train_teacher.py` | RGB + event teacher (trained first) |
| `rgb_baseline.yaml` | `train_rgb_student.py` | RGB-only student |
| `rgb_augmented.yaml` | `train_rgb_student.py` | RGB student with degradation augmentation |
| `naive_distillation.yaml` | `train_distilled_student.py` | Unweighted distillation |
| `object_distillation.yaml` | `train_distilled_student.py` | Object-aware distillation |
| `proposed_event.yaml` | `train_distilled_student.py` | Quality-weighted object-aware distillation |

A config's keys are the command-line options with underscores (`batch_size: 6` is `--batch-size 6`). Options given on the command line override the config, so only machine-specific settings (`--dataset-path`, `--num-workers`) need to be typed. Give every run the same `--num-workers`, because the worker count changes the augmentation draws.

### Event constants

Two constants are measured once on the training split and are already set, so nothing needs measuring before training. The event normalisation constant, `EVENT_SCALE_K = 44`, scales the teacher's event histogram into [0, 1] as `min(log1p(count) / log1p(K), 1)`. It is set in `loaders/dataset_loader.py`, recorded in `docs/event_scale.json`, and measured with `tools/measure_event_scale.py`. The event density reference, `0.4353`, normalises each frame's event coverage for the quality weight, so it only affects `proposed_event`. Training reads it from `docs/event_density.json` at startup, and it is measured with `tools/measure_event_density.py`.

Re-measure only if the training split changes, and only with the complete split:

```bash
python tools/measure_event_scale.py --dataset-path "/path/to/DSEC_ROOT" --all-frames
python tools/measure_event_density.py --dataset-path "/path/to/DSEC_ROOT"
```

- `measure_event_density.py` overwrites `docs/event_density.json`, and the next `proposed_event` run uses the new value. Pass `--output` to write somewhere else.
- `measure_event_scale.py` only prints the value. To use it, edit `EVENT_SCALE_K` in `loaders/dataset_loader.py` and retrain the teacher, since the teacher and all distillation runs must use the same scale.

### Smoke check

Before a full run, check the setup with one epoch on a few frames:

```bash
python train_rgb_student.py --dataset-path "/path/to/DSEC_ROOT" --epochs 1 --max-samples 8 --skip-test
```

It should end with `Test evaluation skipped (--skip-test).` and write `best_student_yolo.pt` to a new `runs/student/exp_NNN/`. Run the same command with `train_teacher.py` to check the event files too.

## 3. Training

### Order

The distilled students need the trained teacher, so train in this order:

**1. Teacher**

```bash
python train_teacher.py --config configs/teacher.yaml --dataset-path "/path/to/DSEC_ROOT" --num-workers 8
```

**2. Make the teacher available to distillation.** The distillation configs read it from `checkpoints/teacher.pt`:

```bash
mkdir -p checkpoints
cp runs/teacher/exp_NNN/best_teacher_yolo.pt checkpoints/teacher.pt
```

**3. RGB students** (these don't need the teacher, so they can run at any point)

```bash
python train_rgb_student.py --config configs/rgb_baseline.yaml  --dataset-path "/path/to/DSEC_ROOT" --num-workers 8
python train_rgb_student.py --config configs/rgb_augmented.yaml --dataset-path "/path/to/DSEC_ROOT" --num-workers 8
```

**4. Distilled students**

```bash
python train_distilled_student.py --config configs/naive_distillation.yaml  --dataset-path "/path/to/DSEC_ROOT" --num-workers 8
python train_distilled_student.py --config configs/object_distillation.yaml --dataset-path "/path/to/DSEC_ROOT" --num-workers 8
python train_distilled_student.py --config configs/proposed_event.yaml      --dataset-path "/path/to/DSEC_ROOT" --num-workers 8
```

Each run validates after every epoch, keeps the checkpoint with the best validation mAP50-95, and finishes with a test evaluation. Add `--skip-test` to stop after training.

### Common options

| Option | Purpose |
| --- | --- |
| `--dataset-path PATH` | DSEC root containing `train/` and `test/` |
| `--num-workers N` | DataLoader workers (keep the same across compared runs) |
| `--epochs N`, `--batch-size N`, `--seed N` | Override the config's values |
| `--max-samples N` | Use N evenly spaced frames per split, for quick checks |
| `--skip-test` | Skip the final test evaluation |
| `--no-augment` | Train on unaugmented frames |
| `--console-level LEVEL` | Console log level (`DEBUG`, `INFO`, `WARNING`, ...) |
| `--teacher-checkpoint PATH` | Teacher to distil from (set by the distillation configs) |

Run `python train_rgb_student.py --help` for the full list.

### Resuming an interrupted run

Each run saves its latest checkpoint after every epoch. To continue in the same folder:

```bash
python train_rgb_student.py --resume runs/student/exp_006
```

Use the script that started the run. Settings come from the original run; only `--num-workers` and `--console-level` may be given again, and `--resume` cannot be combined with `--config`.

### Outputs

Each run creates `runs/<role>/exp_NNN/`, where `<role>` is `teacher`, `student` or `distill`:

- `train.log`: configuration, progress and per-epoch metrics
- `config.yaml`, `run_args.yaml`, `run_command.txt`: the settings used (read by `--resume`)
- `results.csv`, `results.png`: per-epoch metrics and training curves
- `best_<role>_yolo.pt`: the checkpoint with the highest validation mAP50-95
- `latest_<role>_yolo.pt`: the most recent checkpoint, for resuming
- `quality_weights.csv`: per-image quality weights (`proposed_event` only)
- `test/`: confusion matrix, PR curves and example predictions (absent with `--skip-test`)


