import argparse
import log
import sys
import logging
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
    logger: logging.Logger, 
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
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as exc:
        # Stop here: never silently evaluate the final-epoch or incompatible weights.
        raise RuntimeError(
            f"Could not load the best state_dict checkpoint: {checkpoint_path}"
        ) from exc

    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_images = 0
    logger.info(f"Loaded best checkpoint: {checkpoint_path}")
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
    logger.info(f"Test samples: {total_images} | Test batches: {total_batches}")
    logger.info(f"Average test batch loss: {avg_test_batch_loss:.4f}")
    logger.info(f"Average test per-image loss: {avg_test_image_loss:.4f}")
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

def load_checkpoint(checkpoint_path: str | Path, device: torch.device, model: torch.nn.Module, optimiser: torch.optim.AdamW) -> dict:
    """ Load the checkpoint if it exists, otherwise return default values. """
    if not Path(checkpoint_path).is_file():
        return {'past_epochs': 0, 'latest_val_loss': float('inf')}

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    return {'past_epochs': checkpoint["epoch"], 'latest_val_loss': checkpoint["val_loss"]}

# Get model with lowest val_loss: 0 -> best_epoch & best_val_loss, 1 -> everything
def fetch_best_dict(best_path: str | Path, device: torch.device, model: torch.nn.Module, optimiser: torch.optim.AdamW, mode: int = 0) -> dict:
    """ Load the best checkpoint if it exists, otherwise return default values. """
    if not Path(best_path).is_file():
        return {'best_epoch': 0, 'best_val_loss': float('inf')}

    best_state = torch.load(best_path, map_location=device)

    if mode == 1:
        model.load_state_dict(best_state["model_state_dict"])
        optimiser.load_state_dict(best_state["optimiser_state_dict"])

    return {'best_epoch': best_state["epoch"], 'best_val_loss': best_state["val_loss"]}

def validate_model(
        val_loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        logger: logging.Logger
    ) -> float:
    # Evaluate the model on the validation set without updating weights. 
    model.eval()
    total_loss = 0.0
    logger.info("Starting Validation...")
    with torch.no_grad():
        for batch_idx_step, (imgs, targets) in enumerate(val_loader):
            loss, _ = model(_prepare_yolo_batch(imgs, targets, device))
            step_loss = loss.sum().item()
            total_loss += step_loss
            if batch_idx_step % 20 == 0 and batch_idx_step > 0:
                logger.info(f"Val Step [{batch_idx_step}/{len(val_loader)}] "
                      f"| Current Val Loss: {step_loss:.4f}")
                
    return total_loss / len(val_loader)

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    optimiser: torch.optim.AdamW,
    device: torch.device,
    logger: logging.Logger, 
    checkpoint_path: Path = Path("best_student_yolo.pt"),
    latest_path: Path = Path("latest_student_yolo.pt")
) -> Path:
    """Train the student and save the weights with the lowest validation loss."""
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Training and validation loaders must both be non-empty.")
    
    checkpoint_path = Path(checkpoint_path)
    latest_path = Path(latest_path)
    
    total_epochs = 1 # Change this to desired epoch. Hardcoded right now, but can change later
    
    # Checkpoint logic from the weight-saving branch
    checkpoint_info = load_checkpoint(latest_path, device, model, optimiser)
    past_epochs = checkpoint_info["past_epochs"]
    
    if past_epochs >= total_epochs:
        logger.info("ERROR: No more epochs to train")
        return checkpoint_path
        
    # Get the historical best stats so we only overwrite if we beat the previous best
    best_stats = fetch_best_dict(checkpoint_path, device, model, optimiser, mode=0)
    best_val_loss = best_stats["best_val_loss"]
    saved_this_run = False

    summary_width = 62
    
    for epoch in range(past_epochs, total_epochs):
        
        logger.info(f"\n{'=' * summary_width}")
        logger.info(f"{f'Starting Epoch {epoch + 1}/{total_epochs}':^{summary_width}}")
        logger.info(f"{'=' * summary_width}")

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

            step_width = len(str(len(train_loader)))
            epoch_width = len(str(total_epochs))

            if batch_idx_step % 10 == 0:
                logger.info(
                    f"Epoch [{epoch + 1:>{epoch_width}}/{total_epochs}] "
                    f"| Step [{batch_idx_step:>{step_width}}/{len(train_loader)}] "
                    f"| Batch Loss: {step_loss:>10.4f}"
                )

        avg_train_batch_loss = total_loss / len(train_loader)
        avg_train_image_loss = avg_train_batch_loss / train_loader.batch_size
        
        avg_val_batch_loss = validate_model(val_loader, model, device, logger)
        if not math.isfinite(avg_val_batch_loss):
            raise ValueError("Non-finite validation loss; cannot select a best checkpoint.")
            
        # Weight saving per epoch (dict containing model and optimiser states)
        checkpoint_dict = {
            'epoch': epoch + 1, 
            'model_state_dict': model.state_dict(), 
            'optimiser_state_dict': optimiser.state_dict(), 
            'val_loss': avg_val_batch_loss
        }
        
        # Always save the latest checkpoint to resume training if interrupted
        torch.save(checkpoint_dict, latest_path)

        # Save best model if val_loss is lower than the current best_val_loss
        if avg_val_batch_loss < best_val_loss:
            torch.save(checkpoint_dict, checkpoint_path)
            best_val_loss = avg_val_batch_loss
            saved_this_run = True
            logger.info(f"Saved best checkpoint: {checkpoint_path} | Validation loss: {best_val_loss:.4f}")
            
        avg_val_image_loss = avg_val_batch_loss / val_loader.batch_size

        logger.info(f"\n{'=' * summary_width}")
        logger.info(f"{f'Epoch {epoch}/{total_epochs} Summary':^{summary_width}}")
        logger.info(f"{'-' * summary_width}")
        logger.info(
            f"{'Train':<6} | Batch Loss: {avg_train_batch_loss:>10.4f} "
            f"| Image Loss: {avg_train_image_loss:>10.4f}"
        )
        logger.info(
            f"{'Valid':<6} | Batch Loss: {avg_val_batch_loss:>10.4f} "
            f"| Image Loss: {avg_val_image_loss:>10.4f}"
        )
        logger.info(f"{'=' * summary_width}\n")

    # when the run is higher than the historical best, the code below would not save the run. 
    # commented out in case this was the intended behaviour for later adjustment. 
    # if not saved_this_run:
    #     raise RuntimeError("No best checkpoint was saved during this training run.")
    if not checkpoint_path.is_file(): 
        raise RuntimeError("No best checkpoint was saved during this training run")

    return checkpoint_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate, and test the RGB student model."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="DSEC root with labelled train and test directories.",
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
    parser.add_argument(
        "--test-weights",
        type=Path,
        default=None,
        help="Path to a saved checkpoint. If provided, skips training and only runs the test evaluation.",
    )
    return parser.parse_args()

def main() -> None:
    log_directory = log.create_log_directory()
    args = parse_args()
    
    # Create a differently named logger if we are only testing
    is_test_only = args.test_weights is not None
    logger_name = "test" if is_test_only else "train"
    logger = log.create_logger(logger_name, log_directory)

    log.log_command(sys.argv, log_directory)
    device = select_device()
    model = extract_trainable_model("yolo26n.pt", device=str(device))

    if is_test_only:
        logger.info(f"Test only mode activated. Evaluating weights at: {args.test_weights}")
        test_loader = create_rgb_test_loader(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        evaluate_test_model(test_loader, model, device, logger, checkpoint_path=args.test_weights)
        return 

    train_loader, val_loader = create_rgb_dataloaders(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    optimiser = create_optimiser(model)

    # if possible, move the var total_epochs out of training and add as a configurable later
    config = { 
        'Selected Device': device, 
        'Training Samples': len(train_loader.dataset), 
        'Validation Samples': len(val_loader.dataset), 
        'Batch Size': train_loader.batch_size, 
        'Sampler': type(train_loader.sampler).__name__, 
        'Number of Workers': train_loader.num_workers, 
        'Pin Memory': train_loader.pin_memory, 
        'Persistent Workers': train_loader.persistent_workers
    }
    log.log_model_config(config, log_directory, logger)
    
    best_path = log_directory / "best_student_yolo.pt"
    latest_path = log_directory / "latest_student_yolo.pt"

    checkpoint_path = train_model(
        train_loader, 
        val_loader, 
        model, 
        optimiser, 
        device, 
        logger, 
        checkpoint_path = best_path, 
        latest_path = latest_path
    )
    
    if args.skip_test:
        logger.info("Test evaluation skipped (--skip-test).")
        return
        
    test_loader = create_rgb_test_loader(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    evaluate_test_model(test_loader, model, device, logger, checkpoint_path)
    
if __name__ == "__main__":
    main()