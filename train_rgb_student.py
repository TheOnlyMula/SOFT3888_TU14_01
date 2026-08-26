import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset_loader import DSECFusionDataset, yolo_rgb_collate_fn


def select_device() -> torch.device:
    """Select CUDA when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_rgb_dataloader(
    dataset_path: Path,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    """Create the RGB-only DataLoader required by Issue #9."""
    if not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be between 1 and 32")

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    dataset = DSECFusionDataset(
        dataset_path=dataset_path,
        split="train",
        rgb_only=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=yolo_rgb_collate_fn,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure the RGB student DataLoader."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="DSEC dataset root containing the train directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = select_device()
    dataloader = create_rgb_dataloader(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Selected device: {device}")
    print(f"Dataset samples: {len(dataloader.dataset)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Sampler: {type(dataloader.sampler).__name__}")
    print(f"Number of workers: {dataloader.num_workers}")
    print(f"Pin memory: {dataloader.pin_memory}")
    print(f"Persistent workers: {dataloader.persistent_workers}")


if __name__ == "__main__":
    main()
