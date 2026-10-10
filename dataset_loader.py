import argparse
import math
from numbers import Integral
from functools import lru_cache
from pathlib import Path
import hashlib
import cv2
import h5py
import hdf5plugin  # Required to prevent Blosc decompression error
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, get_worker_info
from dsec_det.dataset import DSECDet
from dsec_det.label import CLASSES, COLORS
from dsec_det.io import extract_from_h5_by_timewindow, yaml_file_to_dict
from overexposure import STRENGTH_LEVELS, overexpose, resolve_strength
from darkening import STRENGTH_RANGES as DARKENING_LEVELS, darken
from augment import augment_sample
import logging
from ultralytics import YOLO


SPLIT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "DSEC-det"
    / "config"
    / "train_val_test_split.yaml"
)

EVENT_WINDOW_US = 50_000

EVENT_SCALE_K = 44.0

DEFAULT_DEGRADATION_SHARES = (0.5, 0.25, 0.25)
DEFAULT_STRENGTH_RANGE = (STRENGTH_LEVELS["mild"], STRENGTH_LEVELS["strong"])
DEFAULT_AUGMENTATION_SEED = 0

# Illumination factor that benchmark strength 1 ("strong") darkens to: darkening's "strong" level
DARKEST_ILLUMINATION = DARKENING_LEVELS["strong"]


def validate_augmentation_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("augmentation seed must be a non-negative integer")
    return int(seed)


def validate_degradation_shares(shares):
    """Validate probabilities ordered as clean, dark, overexposed."""
    shares = tuple(float(value) for value in shares)
    if (len(shares) != 3 or any(not math.isfinite(p) or p < 0 for p in shares)
            or not math.isclose(sum(shares), 1.0, rel_tol=0, abs_tol=1e-8)):
        raise ValueError("degradation shares must be three finite non-negative values summing to 1")
    return shares


def validate_strength_range(strength_range):
    """Validate an inclusive minimum/maximum on the benchmark's [0, 1] scale."""
    values = tuple(float(value) for value in strength_range)
    if (len(values) != 2 or any(not math.isfinite(value) for value in values)
            or not 0 <= values[0] <= values[1] <= 1):
        raise ValueError("strength range must be two finite values with 0 <= min <= max <= 1")
    return values


def augment_rgb_degradation(
    image, shares,
    dark_strength_range=DEFAULT_STRENGTH_RANGE,
    overexposed_strength_range=DEFAULT_STRENGTH_RANGE,
    *, rng: np.random.Generator,
):
    """Sample a fresh condition and uniform degradation strength on each access."""
    condition = rng.choice(("clean", "dark", "overexposed"), p=shares)
    if condition == "dark":
        strength = rng.uniform(*dark_strength_range)
        return apply_dark_degradation(image, strength=strength, seed=int(rng.integers(2**63)))
    if condition == "overexposed":
        strength = rng.uniform(*overexposed_strength_range)
        return overexpose(image, strength)
    return image


def degradation_seed(
    evaluation_seed: int,
    sequence_name: str,
    frame_index: int,
    degradation: str,
    strength: float,
) -> int:
    seed_input = (
        f"{evaluation_seed}|{sequence_name}|{frame_index}|{degradation}|{strength}"
    ).encode("utf-8")

    digest = hashlib.sha256(seed_input).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def apply_dark_degradation(image: np.ndarray, strength: float, seed: int) -> np.ndarray:
    """Darken an RGB uint8 image with `darkening.darken` on the shared benchmark strength scale.

    Strength 0 leaves the image unchanged, strength 1 ("strong") is darkening's own "strong" level, and the illumination falls
    linearly in between, so "mild" (0.35) dims to 72% illumination. `seed` fixes the
    illumination field and sensor noise.
    """
    if strength == 0:
        return image.copy()
    illumination = 1.0 - (1.0 - DARKEST_ILLUMINATION) * strength
    return darken(image, illumination, seed=seed)["image"]


def update_rgb_fingerprint(hasher, images: torch.Tensor) -> None:
    rgb = images[:, :3].detach().cpu().contiguous().numpy()
    hasher.update(rgb.tobytes())


def rgb_loader_fingerprint(loader: DataLoader) -> str:
    hasher = hashlib.sha256()

    for images, _ in loader:
        update_rgb_fingerprint(hasher, images)

    return hasher.hexdigest()


@lru_cache(maxsize=None)
def event_time_offset_us(event_file: str) -> int:
    """Absolute timestamp (microseconds) at which a DSEC event file's recording starts."""
    with h5py.File(event_file, "r") as h5f:
        return int(h5f["t_offset"][()])


def load_causal_events(
    base_dataset: DSECDet, idx: int, window_us: int = EVENT_WINDOW_US
) -> dict:
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
    def __init__(
        self,
        dataset_path,
        split="train",
        rgb_only=False,
        degradation="clean",
        degradation_strength=1.0,
        evaluation_seed=0,
        augmentation_shares=None,
        dark_strength_range=DEFAULT_STRENGTH_RANGE,
        overexposed_strength_range=DEFAULT_STRENGTH_RANGE,
        augmentation_seed=DEFAULT_AUGMENTATION_SEED,
        augment=False,
    ):
        self.augmentation_seed = validate_augmentation_seed(augmentation_seed)
        self._augmentation_rng = None
        self._augmentation_worker_id = None
        self.dark_strength_range = validate_strength_range(dark_strength_range)
        self.overexposed_strength_range = validate_strength_range(overexposed_strength_range)
        self.augmentation_shares = None
        if augmentation_shares is not None:
            if split != "train" or not rgb_only or degradation != "clean":
                raise ValueError("degradation augmentation requires clean RGB-only training data")
            self.augmentation_shares = validate_degradation_shares(augmentation_shares)
        split_config = yaml_file_to_dict(SPLIT_CONFIG_PATH)

        self.base_dataset = DSECDet(
            dataset_path, split=split, sync="back", split_config=split_config
        )
        self.rgb_only = rgb_only

        valid_degradations = {"clean", "dark", "overexposed"}
        if degradation not in valid_degradations:
            raise ValueError(f"degradation must be one of {sorted(valid_degradations)}")

        self.degradation = degradation
        self.degradation_strength = resolve_strength(degradation_strength)
        self.evaluation_seed = evaluation_seed
        # Training-only flip, scale, shift and hue/saturation jitter (see augment.py)
        self.augment = augment

    def __len__(self):
        return len(self.base_dataset)

    def augmentation_rng(self):
        """Own stream per worker; never use evaluation or global RNG state.

        Initialise lazily so workers do not inherit identical generator states.
        Persistent workers keep advancing their streams across epochs.
        """
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        if self._augmentation_rng is None or self._augmentation_worker_id != worker_id:
            self._augmentation_rng = np.random.default_rng(
                np.random.SeedSequence(self.augmentation_seed, spawn_key=(worker_id + 1,))
            )
            self._augmentation_worker_id = worker_id
        return self._augmentation_rng

    def generate_event_histogram(self, events, height=480, width=640):
        hist = np.zeros((2, height, width), dtype=np.float32)

        if events is not None and len(events.get("x", [])) > 0:
            # int64 first: p is uint8 and y uint16, so the flat index would overflow
            x = events["x"].astype(np.int64)
            y = events["y"].astype(np.int64)
            p = events["p"].astype(np.int64)

            counts = np.bincount((p * height + y) * width + x, minlength=2 * height * width)
            hist = counts.reshape(2, height, width).astype(np.float32)

        # Log scaling compresses the heavy tail (a typical active pixel sees ~2 events,
        # the busiest hundreds), then the fixed divisor bounds the channels to [0, 1].
        hist_tensor = torch.log1p(torch.from_numpy(hist)) / math.log1p(EVENT_SCALE_K)
        return hist_tensor.clamp_(max=1.0)

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

        if self.augmentation_shares is not None:
            image = augment_rgb_degradation(
                image, self.augmentation_shares,
                self.dark_strength_range, self.overexposed_strength_range,
                rng=self.augmentation_rng(),
            )
        elif self.degradation == "dark":
            # Only darkening is random, so only it needs a per-frame seed.
            frame_index, _, directory = self.base_dataset.rel_index(idx)
            frame_seed = degradation_seed(
                evaluation_seed=self.evaluation_seed,
                sequence_name=directory.root.name,
                frame_index=frame_index,
                degradation=self.degradation,
                strength=self.degradation_strength,
            )
            image = apply_dark_degradation(
                image, strength=self.degradation_strength, seed=frame_seed
            )
        elif self.degradation == "overexposed":
            image = overexpose(image, self.degradation_strength)

        h, w = image.shape[0], image.shape[1]

        tracks = sample["tracks"]
        classes = np.array([int(box["class_id"]) for box in tracks], dtype=np.int64)
        boxes = np.array(
            [[box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]] for box in tracks],
            dtype=np.float64,
        ).reshape(-1, 4)

        event_tensor = None
        if not self.rgb_only:
            event_tensor = self.generate_event_histogram(
                sample.get("events"), height=h, width=w
            )

        if self.augment:
            # One random draw moves RGB, events and boxes together
            events = None if event_tensor is None else event_tensor.permute(1, 2, 0).numpy()
            image, events, boxes, classes = augment_sample(image, events, boxes, classes)
            if events is not None:
                event_tensor = torch.from_numpy(events).permute(2, 0, 1)

        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        yolo_targets = [
            [
                int(cls_id),
                (x1 + x2) / 2.0 / w,
                (y1 + y2) / 2.0 / h,
                (x2 - x1) / w,
                (y2 - y1) / h,
            ]
            for cls_id, (x1, y1, x2, y2) in zip(classes, boxes)
        ]

        target_tensor = torch.tensor(yolo_targets, dtype=torch.float32)

        if self.rgb_only:
            return img_tensor, target_tensor
        else:
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
        targets = torch.cat(targets_with_idx, 0)  # Shape: [Total_Boxes, 6]
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


def _check_loader_settings(
    logger: logging.Logger, batch_size: int, num_workers: int
) -> None:
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
    degradation: str = "clean",
    degradation_strength: float = 1.0,
    evaluation_seed: int = 0,
    augmentation_shares: tuple[float, float, float] | None = None,
    dark_strength_range: tuple[float, float] = DEFAULT_STRENGTH_RANGE,
    overexposed_strength_range: tuple[float, float] = DEFAULT_STRENGTH_RANGE,
    augmentation_seed: int = DEFAULT_AUGMENTATION_SEED,
    augment: bool = False,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible training and validation Dataloaders using explicit sequence splits.

    `rgb_only=True` yields 3-channel RGB images (student); `rgb_only=False` adds the
    2 event-histogram channels (teacher).
    Augmentation shares and strength ranges apply only to training. Validation is
    clean unless `degradation` explicitly requests an evaluation condition; its
    strength and seed are independent of the training augmentation settings.
    `augment` applies geometric and colour augmentation to the train split only. `seed`
    fixes the shuffle order and the per-worker augmentation draws.
    """
    _check_loader_settings(logger, batch_size, num_workers)

    train_dataset = limit_dataset(
        DSECFusionDataset(
            dataset_path=dataset_path, split="train", rgb_only=rgb_only, augment=augment,
            **({
                "augmentation_shares": augmentation_shares,
                "dark_strength_range": dark_strength_range,
                "overexposed_strength_range": overexposed_strength_range,
                "augmentation_seed": augmentation_seed,
            } if augmentation_shares is not None else {}),
        ),
        max_samples,
    )

    val_dataset = limit_dataset(
        DSECFusionDataset(
            dataset_path=dataset_path,
            split="val",
            rgb_only=rgb_only,
            degradation=degradation,
            degradation_strength=degradation_strength,
            evaluation_seed=evaluation_seed,
        ),
        max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=yolo_collate_fn,
        generator=None if seed is None else torch.Generator().manual_seed(seed),
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
    degradation: str = "clean",
    degradation_strength: float = 1.0,
    evaluation_seed: int = 0,
) -> DataLoader:
    """Load clean test images unless an evaluation degradation is explicitly requested.

    Training augmentation settings are deliberately not accepted by this loader.
    """
    _check_loader_settings(logger, batch_size, num_workers)

    test_dataset = DSECFusionDataset(
        dataset_path=dataset_path,
        split="test",
        rgb_only=rgb_only,
        degradation=degradation,
        degradation_strength=degradation_strength,
        evaluation_seed=evaluation_seed,
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


def create_rgb_dataloaders(
    dataset_path: Path, logger: logging.Logger, **kwargs
) -> tuple[DataLoader, DataLoader]:
    """RGB only training and validation Dataloaders for the student."""
    return create_dataloaders(dataset_path, logger, rgb_only=True, **kwargs)


def create_rgb_test_loader(
    dataset_path: Path, logger: logging.Logger, **kwargs
) -> DataLoader:
    """RGB only test DataLoader for the student."""
    return create_test_loader(dataset_path, logger, rgb_only=True, **kwargs)
