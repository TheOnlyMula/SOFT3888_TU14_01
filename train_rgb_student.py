import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, random_split

from ultralytics import YOLO

from dataset_loader import DSECFusionDataset, yolo_rgb_collate_fn


def _prepare_yolo_batch(imgs, targets, device):
    """Move a collated batch to the device and expose YOLO's target fields."""
    imgs = imgs.to(device)
    targets = targets.to(device)
    return {
        "img": imgs,
        "cls": targets[:, 1:2],
        "bboxes": targets[:, 2:],
        "batch_idx": targets[:, 0],
    }


def select_device() -> torch.device:
    """Select CUDA when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_rgb_dataloaders(
    dataset_path: Path,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible 80/20 training and validation Dataloaders."""
    if not 1 <= batch_size <= 50:
        raise ValueError("batch_size must be between 1 and 50")

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    dataset = DSECFusionDataset(
        dataset_path=dataset_path,
        split="train",
        rgb_only=True,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=yolo_rgb_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=yolo_rgb_collate_fn,
    )

    return train_loader, val_loader

def create_rgb_test_loader(
    dataset_path: Path,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    """Load the dedicated labelled test split without shuffling or dropping images."""
    if not 1 <= batch_size <= 50:
        raise ValueError("batch_size must be between 1 and 50")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    test_dataset = DSECFusionDataset(
        dataset_path=dataset_path,
        split="test",
        rgb_only=True,
    )
    if len(test_dataset) == 0:
        raise ValueError("The test split is empty; provide a separate labelled test dataset.")

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=yolo_rgb_collate_fn,
        drop_last=False,
    )


def evaluate_test_model(
    test_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    checkpoint_path: Path = Path("best_student_yolo.pt"),
) -> tuple[float, float]:
    """Reload a trusted state_dict checkpoint and report batch-normalised test loss."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}. Test evaluation was not run."
        )
    if len(test_loader) == 0:
        raise ValueError("The test loader is empty; no test loss can be calculated.")

    model.to(device)
    try:
        state_dict = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        # Stop here: never silently evaluate the final-epoch or incompatible weights.
        raise RuntimeError(
            f"Could not load the best state_dict checkpoint: {checkpoint_path}"
        ) from exc

    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_images = 0
    print(f"Loaded best checkpoint: {checkpoint_path}")
    with torch.no_grad():
        for imgs, targets in test_loader:
            loss, _ = model(_prepare_yolo_batch(imgs, targets, device))
            batch_loss = loss.sum().item()
            if not math.isfinite(batch_loss):
                raise ValueError("Non-finite test loss; no valid test summary is available.")
            total_loss += batch_loss
            total_batches += 1
            total_images += imgs.size(0)

    if total_batches == 0 or total_images == 0:
        raise ValueError("No test images were evaluated.")

    avg_test_batch_loss = total_loss / total_batches
    avg_test_image_loss = total_loss / total_images
    print(f"Test samples: {total_images} | Test batches: {total_batches}")
    print(f"Average test batch loss: {avg_test_batch_loss:.4f}")
    print(f"Average test per-image loss: {avg_test_image_loss:.4f}")
    return avg_test_batch_loss, avg_test_image_loss


def extract_trainable_model(weights_path: str = "yolo26n.pt", device: str | None = None) -> torch.nn.Module:
    """
    Load a pretrained YOLO wrapper, extract the raw PyTorch nn.Module,
    prep it for custom training, and bind loss hyperparameters.

    Returns:
        model (torch.nn.Module): the raw detection model, ready to be
        used directly in a custom PyTorch training loop.
    """
    yolo_wrapper = YOLO(weights_path)
    model = yolo_wrapper.model
    model.train()
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    model.args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
        device=device,
    )

    return model


def create_optimiser(model: torch.nn.Module) -> torch.optim.AdamW:
    """Create the AdamW optimiser for the unfrozen RGB student parameters."""
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
        weight_decay=0.0005,
    )

def validate_model(
    val_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on the validation set without updating weights."""
    model.eval()
    total_loss = 0.0
    print("    Starting Validation...")
    with torch.no_grad():
        for batch_idx_step, (imgs, targets) in enumerate(val_loader):
            loss, _ = model(_prepare_yolo_batch(imgs, targets, device))
            step_loss = loss.sum().item()
            total_loss += step_loss
            if batch_idx_step % 20 == 0 and batch_idx_step > 0:
                print(f"        Val Step [{batch_idx_step}/{len(val_loader)}] "
                      f"| Current Val Loss: {step_loss:.4f}")
    return total_loss / len(val_loader)

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    optimiser: torch.optim.AdamW,
    device: torch.device,
    checkpoint_path: Path = Path("best_student_yolo.pt"),
) -> Path:
    """Train the student and save the weights with the lowest validation loss."""
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Training and validation loaders must both be non-empty.")
    checkpoint_path = Path(checkpoint_path)
    best_val_loss = float("inf")
    saved_this_run = False
    epochs = 1 # Change this to desired epoch. Hardcoded right now, but can change later
    for epoch in range(epochs):
        print(f"--- Starting Epoch {epoch + 1}/{epochs} ---")
        model.train()
        total_loss = 0.0
        for batch_idx_step, (imgs, targets) in enumerate(train_loader):

            batch_data = _prepare_yolo_batch(imgs, targets, device)
            optimiser.zero_grad()
            loss, _ = model(batch_data)
            batch_loss = loss.sum()
            batch_loss.backward()
            optimiser.step()

            step_loss = batch_loss.item()
            total_loss += step_loss

            if batch_idx_step % 10 == 0:
                print(f"    Epoch [{epoch + 1}/{epochs}] Step [{batch_idx_step}/{len(train_loader)}] "
                      f"| Batch Loss: {step_loss:.4f}")

        avg_train_batch_loss = total_loss / len(train_loader)
        avg_train_image_loss = avg_train_batch_loss / train_loader.batch_size
        
        avg_val_batch_loss = validate_model(val_loader, model, device)
        if not math.isfinite(avg_val_batch_loss):
            raise ValueError("Non-finite validation loss; cannot select a best checkpoint.")
        if avg_val_batch_loss < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path)
            best_val_loss = avg_val_batch_loss
            saved_this_run = True
            print(f"Saved best checkpoint: {checkpoint_path} | Validation loss: {best_val_loss:.4f}")
        avg_val_image_loss = avg_val_batch_loss / val_loader.batch_size
        
        print(f"\n================ Epoch {epoch + 1}/{epochs} Summary ================")
        print(f"  Train | Batch Loss: {avg_train_batch_loss:.4f} | Image Loss: {avg_train_image_loss:.4f}")
        print(f"  Valid | Batch Loss: {avg_val_batch_loss:.4f} | Image Loss: {avg_val_image_loss:.4f}")
        print(f"====================================================\n")

    if not saved_this_run:
        raise RuntimeError("No best checkpoint was saved during this training run.")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate, and test the RGB student model."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="DSEC root with labelled train and test directories (test optional with --skip-test).",
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
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Run training and validation only; do not load the test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    train_loader, val_loader = create_rgb_dataloaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = extract_trainable_model("yolo26n.pt", device=str(device))
    optimiser = create_optimiser(model)

    print(f"Selected device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Sampler: {type(train_loader.sampler).__name__}")
    print(f"Number of workers: {train_loader.num_workers}")
    print(f"Pin memory: {train_loader.pin_memory}")
    print(f"Persistent workers: {train_loader.persistent_workers}")
    checkpoint_path = train_model(train_loader, val_loader, model, optimiser, device)
    if args.skip_test:
        print("Test evaluation skipped (--skip-test).")
        return
    test_loader = create_rgb_test_loader(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    evaluate_test_model(test_loader, model, device, checkpoint_path)

if __name__ == "__main__":
    main()
