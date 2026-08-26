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

Example:

```bash
python train_rgb_student.py --dataset-path "/path/to/DSEC_ROOT" --batch-size 4 --num-workers 4
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
