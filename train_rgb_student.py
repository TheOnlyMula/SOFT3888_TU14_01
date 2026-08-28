import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, random_split

from ultralytics import YOLO

from dataset_loader import DSECFusionDataset, yolo_rgb_collate_fn


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
            imgs = imgs.to(device)
            targets = targets.to(device)

            batch_data = {
                "img": imgs,
                "cls": targets[:, 1:2],
                "bboxes": targets[:, 2:],
                "batch_idx": targets[:, 0],
            }

            loss, _ = model(batch_data)
            step_loss = loss.sum().item()
            total_loss += loss.sum().item()
            if batch_idx_step % 20 == 0 and batch_idx_step > 0:
                print(f"        Val Step [{batch_idx_step}/{len(val_loader)}] "
                      f"| Current Val Loss: {step_loss:.4f}")
    return total_loss / len(val_loader)

def train_model(train_loader: DataLoader, val_loader: DataLoader, model: torch.nn.Module, optimiser: torch.optim.AdamW, device: torch.device) -> None:
    """Trains the RGB only student model."""
    epochs = 2 # Change this to desired epoch. Hardcoded right now, but can change later
    for epoch in range(epochs):
        print(f"--- Starting Epoch {epoch + 1}/{epochs} ---")
        model.train()
        total_loss = 0.0
        for batch_idx_step, (imgs, targets) in enumerate(train_loader):

            imgs = imgs.to(device)
            targets = targets.to(device)

            batch_data = {
                "img": imgs,
                "cls": targets[:, 1:2],     # Class IDs
                "bboxes": targets[:, 2:],   # Bounding boxes [x, y, w, h]
                "batch_idx": targets[:, 0]  # Batch indices
            }

            optimiser.zero_grad()

            loss, loss_items = model(batch_data) # loss_items can later be used for debugging if we need

            loss.sum().backward()
            optimiser.step()

            step_loss = loss.sum().item()
            total_loss += step_loss

            if batch_idx_step % 10 == 0:
                print(f"    Epoch [{epoch + 1}/{epochs}] Step [{batch_idx_step}/{len(train_loader)}] "
                      f"| Batch Loss: {step_loss:.4f}")

        avg_train_batch_loss = total_loss / len(train_loader)
        avg_train_image_loss = avg_train_batch_loss / train_loader.batch_size
        
        avg_val_batch_loss = validate_model(val_loader, model, device)
        avg_val_image_loss = avg_val_batch_loss / val_loader.batch_size
        
        print(f"\n================ Epoch {epoch + 1}/{epochs} Summary ================")
        print(f"  Train | Batch Loss: {avg_train_batch_loss:.4f} | Image Loss: {avg_train_image_loss:.4f}")
        print(f"  Valid | Batch Loss: {avg_val_batch_loss:.4f} | Image Loss: {avg_val_image_loss:.4f}")
        print(f"====================================================\n")




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
    train_model(train_loader, val_loader, model, optimiser, device)

if __name__ == "__main__":
    main()
