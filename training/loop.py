"""The training loop: epochs, gradient accumulation, validation and best-checkpoint selection."""
import logging
import math
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from ultralytics.utils.torch_utils import ModelEMA

import log
from evaluate import validate_model, _prepare_yolo_batch
from utils import (
    NOMINAL_BATCH_SIZE,
    accumulation_steps,
    load_checkpoint,
    fetch_best_dict,
    set_loss_schedule,
)

from .losses import batch_losses
from .reporting import format_duration, log_epoch_summary, quality_weight_stats, results_row
from .schedule import WARMUP_EPOCHS, apply_warmup, lr_factor


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
    total_epochs: int,
    checkpoint_path: Path,
    latest_path: Path,
    teacher: torch.nn.Module | None = None,
    lambda_distill: float = 0.0,
    distill_mode: str = "unweighted",
    nominal_batch_size: int = NOMINAL_BATCH_SIZE,
    warmup_epochs: float = WARMUP_EPOCHS,
) -> Path:
    """Train a detector and save the weights with the highest validation mAP50-95.

    Validation loss is not used for selection: the end-to-end loss schedule shifts weight from
    the one-to-many to the one-to-one head each epoch, so loss values are not comparable across
    epochs.

    The optimisation follows Ultralytics' trainer: the summed (not averaged) loss is
    back-propagated, gradients build up over about `nominal_batch_size` images per optimiser
    step, and learning rate, momentum and accumulation warm up over `warmup_epochs` before a
    linear learning-rate decay.

    With a `teacher`, each batch carries RGB + event channels: the student trains on the RGB
    channels while the frozen teacher sees all of them, and `lambda_distill` weights the
    distillation term added to the detection loss. `distill_mode` "unweighted" compares features
    everywhere; "object" only inside each image's enlarged labelled boxes. "object-quality" is
    "object" with each image's loss scaled by its quality weight w = clip(s_e * c_t, 0, 1), so
    the batch distillation loss is the mean of w_i * L_distill_i; each epoch then logs the mean,
    minimum and maximum w and the fraction of images with w = 0.
    """
    if len(train_loader) == 0 or len(val_loader) == 0:
        message = "Training and validation loaders must both be non-empty."
        logger.error(message)
        raise ValueError(message)

    checkpoint_path = Path(checkpoint_path)
    latest_path = Path(latest_path)

    ema = ModelEMA(model)
    # A closed-form schedule, so warmup can rescale the learning rate within an epoch without
    # disturbing the next epoch's value
    scheduler = LambdaLR(optimiser, lambda epoch: lr_factor(epoch, total_epochs))

    checkpoint_info = load_checkpoint(latest_path, device, model, optimiser, ema, scheduler)
    past_epochs = checkpoint_info["past_epochs"]
    if past_epochs:
        logger.info("Resumed from %s after epoch %d", latest_path, past_epochs)

    if past_epochs >= total_epochs:
        logger.info(
            "Training skipped completed_epochs = %d target_epochs = %d",
            past_epochs,
            total_epochs,
        )

    # Get the historical best stats so we only overwrite if we beat the previous best
    best_stats = fetch_best_dict(checkpoint_path, device, model, optimiser, mode = 0)
    best_map50_95 = best_stats["best_map50_95"]

    num_batches = len(train_loader)
    log_every = max(1, num_batches // 20)
    batch_size = train_loader.batch_size
    full_accumulate = accumulation_steps(batch_size, nominal_batch_size)
    # Like Ultralytics, always leave at least the final epoch out of the warmup
    warmup_batches = round(min(warmup_epochs, max(total_epochs - 1, 0)) * num_batches)
    logger.info(
        "Optimiser steps every %d batches (%d images) after a warmup of %d batches",
        full_accumulate,
        full_accumulate * batch_size,
        warmup_batches,
    )

    # Gradients are cleared once here and after each step; like Ultralytics, a partly built-up
    # step carries over into the next epoch.
    optimiser.zero_grad()
    # On resume, start accumulating afresh at the first batch of the next epoch
    last_step_batch = past_epochs * num_batches - 1

    for epoch in range(past_epochs, total_epochs):

        # Shift the end-to-end loss weighting towards the one-to-one head as training progresses
        set_loss_schedule(model, epoch)
        set_loss_schedule(ema.ema, epoch)

        logger.info(
            "Epoch started epoch = %d/%d lr = %.2e",
            epoch + 1,
            total_epochs,
            min(group["lr"] for group in optimiser.param_groups),
        )

        model.train()
        totals = {"loss": 0.0, "box_loss": 0.0, "cls_loss": 0.0, "l1_loss": 0.0}
        if teacher is not None:
            totals["distill_loss"] = 0.0
        quality_weights = []  # Every training image's w this epoch ("object-quality" mode only)
        images_seen = 0
        grad_norm = float("nan")
        epoch_start = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for step, (imgs, targets) in enumerate(train_loader, start=1):

            batch_index = epoch * num_batches + step - 1
            accumulate = full_accumulate
            if batch_index < warmup_batches:
                accumulate = apply_warmup(
                    optimiser,
                    batch_index,
                    warmup_batches,
                    lr_factor(epoch, total_epochs),
                    nominal_batch_size / batch_size,
                )

            batch_data = _prepare_yolo_batch(imgs, targets, device)
            loss, loss_items, distill_loss, weight = batch_losses(model, batch_data, teacher, distill_mode)
            if weight is not None:
                quality_weights.append(weight)

            total_loss = loss.sum()
            if distill_loss is not None:
                total_loss = total_loss + lambda_distill * distill_loss * imgs.shape[0]
            if not torch.isfinite(total_loss):
                message = f"Non-finite training loss at epoch {epoch + 1} step {step}."
                logger.error(message)
                raise ValueError(message)
            total_loss.backward()

            if batch_index - last_step_batch >= accumulate:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0))
                optimiser.step()
                optimiser.zero_grad()
                ema.update(model)
                last_step_batch = batch_index

            batch_loss = total_loss.detach() / imgs.shape[0]
            step_losses = {"loss": batch_loss.item()}
            step_losses.update({name: float(loss_items[name]) for name in ("box_loss", "cls_loss", "l1_loss")})
            if distill_loss is not None:
                step_losses["distill_loss"] = float(distill_loss)
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
                    "distill_loss = %.4f grad_norm = %.2f boxes = %d",
                    step,
                    step_losses["box_loss"],
                    step_losses["cls_loss"],
                    step_losses["l1_loss"],
                    step_losses.get("distill_loss", 0.0),
                    grad_norm,
                    targets.shape[0],
                )

        # The base rate the epoch ended on (the classification head's groups run at a multiple of it)
        epoch_lr = min(group["lr"] for group in optimiser.param_groups)
        scheduler.step()
        train_time = time.monotonic() - epoch_start
        train_results = {name: total / num_batches for name, total in totals.items()}
        weight_stats = quality_weight_stats(quality_weights)

        # Validate using the EMA model for smooth, stable metrics
        val_start = time.monotonic()
        val_results = validate_model(val_loader, ema.ema, device, logger)
        val_time = time.monotonic() - val_start
        avg_val_loss = val_results["loss"]
        val_map50_95 = val_results["map50_95"]

        if not math.isfinite(avg_val_loss):
            message = "Non-finite validation loss; cannot select a best checkpoint."
            logger.error(message)
            raise ValueError(message)
        if not math.isfinite(val_map50_95):
            message = "Non-finite validation mAP50-95; cannot select a best checkpoint."
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

        # Save best model if mAP50-95 beats the current best; ties keep the earlier epoch
        is_best = val_map50_95 > best_map50_95
        if is_best:
            torch.save(checkpoint_dict, checkpoint_path)
            logger.info(
                "New best weights saved epoch = %d mAP50-95 = %.4f (previous best = %.4f) path = %s",
                epoch + 1,
                val_map50_95,
                best_map50_95,
                checkpoint_path,
            )
            best_map50_95 = val_map50_95

        logger.info(
            "Epoch completed epoch = %d/%d train_time = %s val_time = %s best_checkpoint = %s",
            epoch + 1,
            total_epochs,
            format_duration(train_time),
            format_duration(val_time),
            "saved" if is_best else "unchanged (best mAP50-95 = %.4f)" % best_map50_95,
        )
        log_epoch_summary(logger, train_results, val_results, weight_stats, distill_mode, lambda_distill)
        row = results_row(epoch + 1, train_results, val_results, weight_stats, epoch_lr)
        log.log_epoch_results(row, checkpoint_path.parent)

    if not checkpoint_path.is_file():
        message = "No best checkpoint was saved during this run."
        logger.error(message)
        raise RuntimeError(message)
    return checkpoint_path
