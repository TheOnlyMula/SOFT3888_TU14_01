import argparse
from functools import lru_cache
from pathlib import Path
import cv2
import h5py
import hdf5plugin  # Required to prevent Blosc decompression error
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from dsec_det.dataset import DSECDet
from dsec_det.label import CLASSES, COLORS
from dsec_det.io import extract_from_h5_by_timewindow, yaml_file_to_dict
import logging
from ultralytics import YOLO 


SPLIT_CONFIG_PATH = Path(__file__).resolve().parent / "DSEC-det" / "config" / "train_val_test_split.yaml"

EVENT_WINDOW_US = 50_000


@lru_cache(maxsize=None)
def event_time_offset_us(event_file: str) -> int:
    """Absolute timestamp (microseconds) at which a DSEC event file's recording starts."""
    with h5py.File(event_file, "r") as h5f:
        return int(h5f["t_offset"][()])


def load_causal_events(base_dataset: DSECDet, idx: int, window_us: int = EVENT_WINDOW_US) -> dict:
    """Events from the `window_us` microseconds up to the image timestamp, never after it.

    Real-time detection only has past events when an image arrives. DSECDet's own event windows
    do not fit: sync="back" returns the 50 ms *after* the image, and sync="front" does return the
    50 ms before it but pairs the image with the previous frame's labels. So images and labels come
    from DSECDet (sync="back", which keeps them aligned) and the event window is built here.
    At the start of a recording the window is clipped, so the first frame of a sequence has few or no events.
    """
    index, _, directory = base_dataset.rel_index(idx)
    t_image = int(directory.images.timestamps[index])
    event_file = directory.events.event_file
    t_start = max(t_image - window_us, event_time_offset_us(str(event_file)))
    return extract_from_h5_by_timewindow(event_file, t_start, t_image)


class DSECFusionDataset(Dataset):
    def __init__(self, dataset_path, split="train", rgb_only=False): 
        split_config = yaml_file_to_dict(SPLIT_CONFIG_PATH)

        self.base_dataset = DSECDet(
            dataset_path, 
            split=split, 
            sync="back", 
            split_config=split_config
        )
        self.rgb_only = rgb_only

    def __len__(self):
        return len(self.base_dataset)

    def generate_event_histogram(self, events, height=480, width=640):        
        hist = np.zeros((2, height, width), dtype=np.float32)

        if events is not None and len(events.get("x", [])) > 0:
            x = events["x"].astype(int)
            y = events["y"].astype(int)
            p = events["p"].astype(int)

            # Channel 0 (Negative/0), Channel 1 (Positive/1)
            np.add.at(hist[0], (y[p == 0], x[p == 0]), 1)
            np.add.at(hist[1], (y[p == 1], x[p == 1]), 1)

        hist_tensor = torch.from_numpy(hist)
        # Apply Log Scaling normalization
        hist_tensor = torch.log1p(hist_tensor)
        return hist_tensor

    def __getitem__(self, idx):
        # RGB-only training/testing must not open event files.
        sample = {
            "image": self.base_dataset.get_image(idx),
            "tracks": self.base_dataset.get_tracks(idx),
        }
        if not self.rgb_only:
            sample["events"] = load_causal_events(self.base_dataset, idx)

        # RGB Image: cv2 loads BGR, but the pretrained YOLO weights expect RGB
        image = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB)
        h, w = image.shape[0], image.shape[1]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # YOLO Bounding Box Conversion
        yolo_targets = []
        for box in sample["tracks"]:
            cls_id = int(box["class_id"])
            box_x, box_y, box_w, box_h = box["x"], box["y"], box["w"], box["h"]
            
            # Calculate center points and normalize
            x_center = (box_x + box_w / 2.0) / w
            y_center = (box_y + box_h / 2.0) / h
            norm_w = box_w / w
            norm_h = box_h / h
            
            yolo_targets.append([cls_id, x_center, y_center, norm_w, norm_h])
            
        # Convert to a PyTorch tensor
        target_tensor = torch.tensor(yolo_targets, dtype=torch.float32)

        if self.rgb_only:
            return img_tensor, target_tensor
        else:
            event_tensor = self.generate_event_histogram(
                sample.get("events"), height=h, width=w
            )
            fused_tensor = torch.cat((img_tensor, event_tensor), dim=0)
            return fused_tensor, target_tensor

def yolo_collate_fn(batch):
    """Collate function for RGB-only YOLO training."""
    imgs, targets = zip(*batch)
    
    # Stack images into a standard batch tensor
    imgs = torch.stack(imgs, 0)
    
    # Add the batch index to the targets
    targets_with_idx = []
    for i, target in enumerate(targets):
        if target.shape[0] > 0:
            batch_idx = torch.full((target.shape[0], 1), i, dtype=target.dtype)
            targets_with_idx.append(torch.cat((batch_idx, target), dim=1))
            
    if len(targets_with_idx) > 0:
        targets = torch.cat(targets_with_idx, 0) # Shape: [Total_Boxes, 6]
    else:
        targets = torch.empty((0, 6))
        
    return imgs, targets


def limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    step = len(dataset) / max_samples
    return Subset(dataset, [int(i * step) for i in range(max_samples)])


def sequence_names(dataset: Dataset) -> list[str]:
    indices = dataset.indices if isinstance(dataset, Subset) else None
    base = (dataset.dataset if isinstance(dataset, Subset) else dataset).base_dataset
    names, start = [], 0
    for directory in base.subsequence_directories:
        count = len(base.img_idx_track_idxs[directory.name]) - 1
        if indices is None or any(start <= i < start + count for i in indices):
            names.append(directory.name)
        start += count
    return names


def _check_loader_settings(logger: logging.Logger, batch_size: int, num_workers: int) -> None:
    if not 1 <= batch_size <= 32:
        message = "batch_size must be between 1 and 32"
        logger.error(message)
        raise ValueError(message)

    if num_workers < 0:
        message = "num_workers must be non-negative"
        logger.error(message)
        raise ValueError(message)


def create_dataloaders(
    dataset_path: Path,
    logger: logging.Logger,
    rgb_only: bool,
    batch_size: int = 4,
    num_workers: int = 0,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible training and validation Dataloaders using explicit sequence splits.

    `rgb_only=True` yields 3-channel RGB images (student); `rgb_only=False` adds the
    2 event-histogram channels (teacher).
    """
    _check_loader_settings(logger, batch_size, num_workers)

    train_dataset = limit_dataset(DSECFusionDataset(
        dataset_path=dataset_path,
        split="train",
        rgb_only=rgb_only
    ), max_samples)

    val_dataset = limit_dataset(DSECFusionDataset(
        dataset_path=dataset_path,
        split="val",
        rgb_only=rgb_only
    ), max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=yolo_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=yolo_collate_fn,
    )

    return train_loader, val_loader


def create_test_loader(
    dataset_path: Path,
    logger: logging.Logger,
    rgb_only: bool,
    batch_size: int = 4,
    num_workers: int = 0,
    max_samples: int | None = None,
) -> DataLoader:
    _check_loader_settings(logger, batch_size, num_workers)

    test_dataset = DSECFusionDataset(
        dataset_path=dataset_path,
        split="test",
        rgb_only=rgb_only,
    )
    if len(test_dataset) == 0:
        message = "The test split is empty; provide a labelled test dataset."
        logger.error(message)
        raise ValueError(message)

    return DataLoader(
        limit_dataset(test_dataset, max_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=yolo_collate_fn,
        drop_last=False,
    )


def create_rgb_dataloaders(dataset_path: Path, logger: logging.Logger, **kwargs) -> tuple[DataLoader, DataLoader]:
    """RGB only training and validation Dataloaders for the student."""
    return create_dataloaders(dataset_path, logger, rgb_only=True, **kwargs)


def create_rgb_test_loader(dataset_path: Path, logger: logging.Logger, **kwargs) -> DataLoader:
    """RGB only test DataLoader for the student."""
    return create_test_loader(dataset_path, logger, rgb_only=True, **kwargs)
