"""Shared training entry point for the RGB student and the RGB + event teacher.

Both models use the same YOLO26n architecture, training loop, validation, logging and test
evaluation. The only difference is the input: the student sees 3 RGB channels, the teacher
sees RGB plus a 2-channel event histogram (5 channels).
"""
import argparse
import log
import sys
import logging
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from evaluate import validate_model, evaluate_test_model, _prepare_yolo_batch
from dataset_loader import create_dataloaders, create_test_loader, sequence_names
from ultralytics.utils.torch_utils import ModelEMA
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import (
    select_device,
    extract_trainable_model,
    create_optimiser,
    load_checkpoint,
    fetch_best_dict,
    set_loss_schedule,
)

def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes}m{seconds:02d}s"


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
    total_epochs: int,
    checkpoint_path: Path,
    latest_path: Path
) -> Path:
    """Train a detector and save the weights with the lowest validation loss."""
    if len(train_loader) == 0 or len(val_loader) == 0:
        message = "Training and validation loaders must both be non-empty."
        logger.error(message)
        raise ValueError(message)

    checkpoint_path = Path(checkpoint_path)
    latest_path = Path(latest_path)

    ema = ModelEMA(model)
    scheduler = CosineAnnealingLR(optimiser, T_max=total_epochs, eta_min=1e-6)

    checkpoint_info = load_checkpoint(latest_path, device, model, optimiser, ema, scheduler)
    past_epochs = checkpoint_info["past_epochs"]

    if past_epochs >= total_epochs:
        logger.info(
            "Training skipped completed_epochs = %d target_epochs = %d",
            past_epochs,
            total_epochs,
        )

    # Get the historical best stats so we only overwrite if we beat the previous best
    best_stats = fetch_best_dict(checkpoint_path, device, model, optimiser, mode = 0)
    best_val_loss = best_stats["best_val_loss"]

    num_batches = len(train_loader)
    log_every = max(1, num_batches // 20)

    for epoch in range(past_epochs, total_epochs):

        # Shift the end-to-end loss weighting towards the one-to-one head as training progresses
        set_loss_schedule(model, epoch)
        set_loss_schedule(ema.ema, epoch)

        logger.info(
            "Epoch started epoch = %d/%d lr = %.2e",
            epoch + 1,
            total_epochs,
            optimiser.param_groups[0]["lr"],
        )

        model.train()
        totals = {"loss": 0.0, "box_loss": 0.0, "cls_loss": 0.0, "l1_loss": 0.0}
        images_seen = 0
        epoch_start = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for step, (imgs, targets) in enumerate(train_loader, start=1):

            batch_data = _prepare_yolo_batch(imgs, targets, device)
            optimiser.zero_grad()

            loss, loss_items = model(batch_data)
            batch_loss = loss.sum() / imgs.shape[0]
            if not torch.isfinite(batch_loss):
                message = f"Non-finite training loss at epoch {epoch + 1} step {step}."
                logger.error(message)
                raise ValueError(message)
            batch_loss.backward()

            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0))

            optimiser.step()
            ema.update(model)

            step_losses = {"loss": batch_loss.item()}
            step_losses.update({name: float(loss_items[name]) for name in ("box_loss", "cls_loss", "l1_loss")})
            for name, value in step_losses.items():
                totals[name] += value
            images_seen += imgs.shape[0]

            if step == 1 or step % log_every == 0 or step == num_batches:
                elapsed = time.monotonic() - epoch_start
                images_per_second = images_seen / elapsed if elapsed > 0 else 0.0
                eta = elapsed / step * (num_batches - step)
                gpu_memory = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
                logger.info(
                    "Batch completed epoch = %d/%d step = %d/%d loss = %.4f avg_loss = %.4f "
                    "img/s = %.1f eta = %s gpu_mem = %.1fGB",
                    epoch + 1,
                    total_epochs,
                    step,
                    num_batches,
                    step_losses["loss"],
                    totals["loss"] / step,
                    images_per_second,
                    format_duration(eta),
                    gpu_memory,
                )
                logger.debug(
                    "Batch details step = %d box_loss = %.4f cls_loss = %.4f l1_loss = %.4f "
                    "grad_norm = %.2f boxes = %d",
                    step,
                    step_losses["box_loss"],
                    step_losses["cls_loss"],
                    step_losses["l1_loss"],
                    grad_norm,
                    targets.shape[0],
                )

        scheduler.step()
        train_time = time.monotonic() - epoch_start
        train_results = {name: total / num_batches for name, total in totals.items()}

        # Validate using the EMA model for smooth, stable metrics
        val_start = time.monotonic()
        val_results = validate_model(val_loader, ema.ema, device, logger)
        val_time = time.monotonic() - val_start
        avg_val_loss = val_results["loss"]

        if not math.isfinite(avg_val_loss):
            message = "Non-finite validation loss; cannot select a best checkpoint."
            logger.error(message)
            raise ValueError(message)

        # Weight saving per epoch
        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': ema.ema.state_dict(),
            'raw_model_state_dict': model.state_dict(),
            'ema_updates': ema.updates,
            'optimiser_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'val_metrics': val_results,
        }

        # Always save the latest checkpoint to resume training if interrupted
        torch.save(checkpoint_dict, latest_path)

        # Save best model if val_loss is lower than the current best_val_loss
        is_best = avg_val_loss < best_val_loss
        if is_best:
            torch.save(checkpoint_dict, checkpoint_path)
            best_val_loss = avg_val_loss

        logger.info(
            "Epoch completed epoch = %d/%d train_time = %s val_time = %s best_checkpoint = %s",
            epoch + 1,
            total_epochs,
            format_duration(train_time),
            format_duration(val_time),
            "saved" if is_best else "unchanged (best val_loss = %.4f)" % best_val_loss,
        )
        for split, results in (("train", train_results), ("val", val_results)):
            logger.info(
                "  %-5s loss = %.4f box_loss = %.4f cls_loss = %.4f l1_loss = %.4f",
                split,
                results["loss"],
                results["box_loss"],
                results["cls_loss"],
                results["l1_loss"],
            )
        logger.info(
            "  val   precision = %.4f recall = %.4f mAP50 = %.4f mAP50-95 = %.4f agnostic_mAP50 = %.4f",
            val_results["precision"],
            val_results["recall"],
            val_results["map50"],
            val_results["map50_95"],
            val_results["agnostic_map50"],
        )

    if not checkpoint_path.is_file():
        message = "No best checkpoint was saved during this run."
        logger.error(message)
        raise RuntimeError(message)
    return checkpoint_path

def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset-path", type=Path, required=True,
        help="DSEC root with labelled train and val directories.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1, help="Total number of epochs to train.")
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Run training and validation only; do not run final eval step.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit each of train/val/test to this many evenly spaced frames (for quick smoke runs).",
    )
    parser.add_argument(
        "--console-level", type = str, default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help = "Set the logging level.",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    return args

def main(role: str, rgb_only: bool) -> None:
    """Train, validate and test one model. `role` is "student" or "teacher"."""
    input_channels = 3 if rgb_only else 5
    description = (
        "Train and validate the RGB student model." if rgb_only
        else "Train and validate the RGB + event teacher model."
    )
    args = parse_args(description)

    log_directory = log.create_log_directory(role)

    logger = log.create_logger(
        "train",
        log_directory,
        console_level=args.console_level,
    )
    log.log_command(sys.argv, log_directory)

    device = select_device()

    model = extract_trainable_model(
        ch=input_channels,
        weights_path="yolo26n.pt",
        device=device,
        epochs=args.epochs,
    )

    train_loader, val_loader = create_dataloaders(
        dataset_path = args.dataset_path,
        logger = logger,
        rgb_only = rgb_only,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        max_samples = args.max_samples,
    )

    optimiser = create_optimiser(model)

    config = {
        'Model': role,
        'Input Channels': input_channels,
        'Selected Device': device,
        'Epochs': args.epochs,
        'Training Samples': len(train_loader.dataset),
        'Validation Samples': len(val_loader.dataset),
        'Batch Size': train_loader.batch_size,
        'Sampler': type(train_loader.sampler).__name__,
        'Number of Workers': train_loader.num_workers,
        'Pin Memory': train_loader.pin_memory,
        'Persistent Workers': train_loader.persistent_workers,
        'Console Level' : args.console_level,
        'Skip Test Evaluation': args.skip_test,
        'Max Samples': args.max_samples,
    }
    if args.max_samples is not None:
        config['Training Sequences'] = sequence_names(train_loader.dataset)
        config['Validation Sequences'] = sequence_names(val_loader.dataset)
    log.log_model_config(config, log_directory, logger)

    best_path = log_directory / f"best_{role}_yolo.pt"
    latest_path = log_directory / f"latest_{role}_yolo.pt"

    checkpoint_path = train_model(
        train_loader,
        val_loader,
        model,
        optimiser,
        device,
        logger,
        total_epochs=args.epochs,
        checkpoint_path=best_path,
        latest_path=latest_path
    )

    if args.skip_test:
        logger.info("Test evaluation skipped (--skip-test).")
        return

    test_loader = create_test_loader(
        dataset_path=args.dataset_path,
        logger=logger,
        rgb_only=rgb_only,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    if args.max_samples is not None:
        logger.info("Test samples = %d sequences = %s", len(test_loader.dataset), sequence_names(test_loader.dataset))
    evaluate_test_model(
        test_loader=test_loader,
        model=model,
        device=device,
        logger=logger,
        checkpoint_path=checkpoint_path,
        save_dir=log_directory / "test",
    )
