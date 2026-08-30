# SOFT3888_TU14_01

## 📚 Project Documentation

All core administrative documents and weekly records for this capstone project are hosted in our repository's **[Wiki](https://github.sydney.edu.au/aalf0384/SOFT3888_TU14_01/wiki)**. 

Please refer to the Wiki for:
* **Core Documents:** Group Contract.
* **Meeting Minutes:** Weekly progress records and action items.

---

## 🛠️ Environment Setup and Installation

Follow these steps to configure your local Python environment to run the project smoothly.

**1. Install Utility Packages**
We rely on the official DSEC-Detection package to handle the raw dataset files. Clone the repository and install it in editable mode:
```bash
git clone https://github.com/uzh-rpg/DSEC-det.git
cd dsec-det/
pip install -e .
```

**2. Install Project Dependencies**
Install the necessary packages for our specific code (such as OpenCV and hdf5plugin) using the requirements file:
```bash
pip install -r requirements.txt
```

## 📊 DSEC Dataset Loader

**Purpose**
Acts as a PyTorch Dataset loader that visualises and returns:

*   **RGB Images:** Traditional camera frames.
*   **Event Data:** Asynchronous microsecond level pixel polarity changes, converted into a 2-channel event histogram (positive and negative events).
*   **Detection Labels:** Bounding box coordinates and classification labels.

The script leverages the core `DSECDet` utility with a `sync="back"` configuration. This strictly bundles the RGB image with the 50 milliseconds of event data that occurred *immediately prior* to the camera shutter, satisfying the causal timeline required..

**Usage Instructions**
You can run the script directly from your terminal to visualize the synchronized RGB image alongside the generated event histogram, complete with bounding box overlays.

```bash
# Basic usage
python dataset_loader.py --dataset-path "/path/to/your/dataset"
```

## RGB Student DataLoader

`train_rgb_student.py` configures the RGB-only student DataLoader for Issue #9.

The dataset path must point to the DSEC root containing the `train` directory:

```text
DSEC_ROOT/
└── train/
    └── sequence_name/
        ├── calibration/
        ├── events/
        ├── images/
        └── object_detections/
```

Training/validation-only example (no test directory required):

```bash
python train_rgb_student.py --dataset-path "/path/to/DSEC_ROOT" --batch-size 4 --num-workers 4 --skip-test
```

The script:

* Selects CUDA automatically when available, otherwise CPU.
* Uses `DSECFusionDataset` with `split="train"` and `rgb_only=True`.
* Enables shuffled loading with `yolo_rgb_collate_fn`.
* Enables pinned memory.
* Enables persistent workers when `num_workers` is greater than zero.

When using `--num-workers 0`, persistent workers are disabled because PyTorch does not permit persistent workers without worker processes.

On Windows, use an ASCII-only dataset path where possible because OpenCV may fail to read images from paths containing non-ASCII characters.

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

## Post-training test evaluation

Training retains the existing seed-42 80/20 train/validation split and two epochs. The model with the lowest validation batch loss is saved as `best_student_yolo.pt`. After training, it is restored and evaluated on `split="test"` with `shuffle=False`, `drop_last=False`, `model.eval()`, and `torch.no_grad()`.

Provide separate labelled `train/` and `test/` directories. Each sequence needs `images/timestamps.txt`, `images/left/distorted/*.png`, and `object_detections/left/tracks.npy`, with aligned filenames, timestamps, and labels. RGB-only loading does not require event files; RGB+event mode still does. Do not use test sequences for training, validation, or model tuning.

Run from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe train_rgb_student.py --dataset-path "D:\DSEC_ROOT" --batch-size 4 --num-workers 0
```

Append `--skip-test` for training/validation ONLY when no test split is available.

The checkpoint is a PyTorch `state_dict` for the same model architecture, not an Ultralytics model archive. Use only trusted checkpoints. Loading failures, empty test data, or nonfinite losses stop evaluation without a final test summary.

The console prints the loaded checkpoint, sample/batch counts, and:

- **Average test batch loss:** total batch loss divided by processed batches.
- **Average test per-image loss:** total batch loss divided by processed images.

Batch loss is `loss.sum().item()`. Actual image counts include a partial last batch. These values are loss summaries, not mAP, precision, or recall.

Install the dependencies in `requirements.txt` and DSEC-Detection in the same environment. Last verified on Windows/CPU with Python 3.14.7; fresh installations and CUDA have not been verified. Run the unit-test command above after changes.

### Small integration check

Use a prepared small labelled subset and a **new** output directory:

```powershell
.\.venv\Scripts\python.exe tools/run_issue17_smoke.py --dataset-root "D:\DSEC_ROOT" --output-dir "D:\DSEC_issue17_verify"
```

The helper uses 16 training and 4 validation samples, then evaluates **every** sample in the supplied test split. It records versions and selected indices in `run_metadata.json`, and measured losses/checkpoint verification in `test_results.json`. Keep datasets, weights, checkpoints, and logs out of Git. This checks integration, not full-dataset performance or generalisation; confirm the team's split provenance before reporting formal results.
