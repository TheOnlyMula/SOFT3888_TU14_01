import argparse
from pathlib import Path
import cv2
import hdf5plugin  # Required to prevent Blosc decompression error
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dsec_det.dataset import DSECDet
from dsec_det.label import CLASSES, COLORS
from dsec_det.io import yaml_file_to_dict
import logging
from ultralytics import YOLO 


class DSECFusionDataset(Dataset):
    def __init__(self, dataset_path, split="train", rgb_only=False): 
        split_config = yaml_file_to_dict(Path("./DSEC-det/config/train_val_test_split.yaml"))

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
        if self.rgb_only:
            # RGB-only training/testing must not open event files.
            sample = {
                "image": self.base_dataset.get_image(idx),
                "tracks": self.base_dataset.get_tracks(idx),
            }
        else:
            sample = self.base_dataset[idx]

        # RGB Image
        image = sample["image"]
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


def create_rgb_dataloaders(
    dataset_path: Path,
    logger: logging.Logger,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible training and validation Dataloaders using explicit sequence splits."""
    if not 1 <= batch_size <= 32:
        message = "batch_size must be between 1 and 32"
        logger.error(message)
        raise ValueError(message)
    
    if num_workers < 0:
        message = "num_workers must be non-negative"
        logger.error(message)
        raise ValueError(message)
        
    train_dataset = DSECFusionDataset(
        dataset_path=dataset_path,
        split="train",
        rgb_only=True
    )

    val_dataset = DSECFusionDataset(
        dataset_path=dataset_path,
        split="val",
        rgb_only=True
    )

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

def create_rgb_test_loader(
    dataset_path: Path,
    logger: logging.Logger,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    """Create the dedicated, sequential RGB-only test DataLoader."""
    if not 1 <= batch_size <= 32:
        message = "batch_size must be between 1 and 32"
        logger.error(message)
        raise ValueError(message)

    if num_workers < 0:
        message = "num_workers must be non-negative"
        logger.error(message)
        raise ValueError(message)

    test_dataset = DSECFusionDataset(
        dataset_path=dataset_path,
        split="test",
        rgb_only=True,
    )
    if len(test_dataset) == 0:
        message = "The test split is empty; provide a labelled test dataset."
        logger.error(message)
        raise ValueError(message)

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=yolo_collate_fn,
        drop_last=False,
    )


def create_fusion_dataloaders(
    dataset_path: Path,
    logger: logging.Logger,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible training and validation Dataloaders for the Fusion Teacher."""
    if not 1 <= batch_size <= 32:
        message = "batch_size must be between 1 and 32"
        logger.error(message)
        raise ValueError(message)
    if num_workers < 0:
        message = "num_workers must be non-negative"
        logger.error(message)
        raise ValueError(message)
        
    train_dataset = DSECFusionDataset(dataset_path=dataset_path, split="train", rgb_only=False)
    val_dataset = DSECFusionDataset(dataset_path=dataset_path, split="val", rgb_only=False)

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

def create_test_dataloader(
    dataset_path: Path,
    rgb_only: bool,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    """Create reproducible test Dataloader using explicit sequence splits."""
    if not 1 <= batch_size <= 50:
        raise ValueError("batch_size must be between 1 and 50")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    
    test_dataset = DSECFusionDataset(dataset_path=dataset_path, split="test", rgb_only=rgb_only)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=yolo_collate_fn, 
    )
    return test_loader