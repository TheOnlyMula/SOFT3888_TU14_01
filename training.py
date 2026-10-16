"""Shared training entry point for the RGB student and the RGB + event teacher.

Both models use the same YOLO26n architecture, training loop, validation, logging and test
evaluation. The only difference is the input: the student sees 3 RGB channels, the teacher
sees RGB plus a 2-channel event histogram (5 channels).

Passing a teacher checkpoint turns on distillation: the student still trains on RGB only, while
a frozen teacher sees the paired RGB + event frames and supervises the student's features.
Validation and testing always use RGB-only data, so student results never depend on events.
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
from dataset_loader import (
    DEFAULT_DEGRADATION_SHARES, DEFAULT_STRENGTH_RANGE, DEFAULT_AUGMENTATION_SEED,
    validate_degradation_shares, validate_strength_range, validate_augmentation_seed,
)
from distillation import (
    DISTILL_MODES,
    MASK_ENLARGEMENT,
    TEACHER_CONFIDENCE_THRESHOLD,
    teacher_confidence,
    distilled_features,
    feature_distillation_loss,
    load_teacher,
    object_distillation_loss,
    object_region_mask,
)
import numpy as np
from ultralytics.utils.torch_utils import ModelEMA, init_seeds
from torch.optim.lr_scheduler import LambdaLR
from utils import (
    NOMINAL_BATCH_SIZE,
    OPTIMISER_LR,
    accumulation_steps,
    select_device,
    extract_trainable_model,
    create_optimiser,
    load_checkpoint,
    fetch_best_dict,
    set_loss_schedule,
)

# Warmup and schedule follow Ultralytics' YOLO26 defaults ("optimizer=auto" picking MuSGD).
# Over the warmup the learning rate rises from 0, momentum rises from WARMUP_MOMENTUM, and the
# images per optimiser step rise from one batch to NOMINAL_BATCH_SIZE.
WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
MOMENTUM_AFTER_WARMUP = 0.937
FINAL_LR_FRACTION = 0.01


def lr_factor(epoch: int, total_epochs: int) -> float:
    """Fraction of the starting learning rate used in `epoch` (Ultralytics' linear schedule)."""
    return max(1 - epoch / total_epochs, 0) * (1 - FINAL_LR_FRACTION) + FINAL_LR_FRACTION


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
    everywhere; "object" only inside each image's enlarged labelled boxes.
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
        images_seen = 0
        grad_norm = float("nan")
        epoch_start = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for step, (imgs, targets) in enumerate(train_loader, start=1):

            batch_index = epoch * num_batches + step - 1
            accumulate = full_accumulate
            if batch_index < warmup_batches:
                ramp = [0, warmup_batches]
                accumulate = max(1, round(float(np.interp(batch_index, ramp, [1, nominal_batch_size / batch_size]))))
                for group in optimiser.param_groups:
                    target_lr = group["initial_lr"] * lr_factor(epoch, total_epochs)
                    group["lr"] = float(np.interp(batch_index, ramp, [0.0, target_lr]))
                    group["momentum"] = float(
                        np.interp(batch_index, ramp, [WARMUP_MOMENTUM, MOMENTUM_AFTER_WARMUP])
                    )

            batch_data = _prepare_yolo_batch(imgs, targets, device)

            if teacher is None:
                loss, loss_items = model(batch_data)
                distill_loss = None
            else:
                # The student sees RGB only; the teacher sees the same frames plus events.
                student_batch = {**batch_data, "img": batch_data["img"][:, :3]}
                student_preds = model(student_batch["img"])
                loss, loss_items = model.loss(student_batch, student_preds)
                with torch.no_grad():
                    teacher_preds = teacher(batch_data["img"])
                    teacher_confidences = teacher_confidence(
                        teacher_preds,
                        threshold=TEACHER_CONFIDENCE_THRESHOLD,
                    )
                if distill_mode == "object":
                    mask = object_region_mask(
                        batch_data["batch_idx"],
                        batch_data["bboxes"],
                        imgs.shape[0],
                        distilled_features(student_preds, teacher_preds)[0].shape[-2:],
                    )
                    distill_loss = object_distillation_loss(student_preds, teacher_preds, mask).mean()
                else:
                    distill_loss = feature_distillation_loss(student_preds, teacher_preds)

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
        for split, results in (("train", train_results), ("val", val_results)):
            logger.info(
                "  %-5s loss = %.4f box_loss = %.4f cls_loss = %.4f l1_loss = %.4f",
                split,
                results["loss"],
                results["box_loss"],
                results["cls_loss"],
                results["l1_loss"],
            )
        if teacher is not None:
            logger.info(
                "  train distill_loss = %.4f (mode = %s, lambda = %.3g, contribution = %.4f)",
                train_results["distill_loss"],
                distill_mode,
                lambda_distill,
                lambda_distill * train_results["distill_loss"],
            )
        logger.info(
            "  val   precision = %.4f recall = %.4f mAP50 = %.4f mAP50-95 = %.4f agnostic_mAP50 = %.4f",
            val_results["precision"],
            val_results["recall"],
            val_results["map50"],
            val_results["map50_95"],
            val_results["agnostic_map50"],
        )

        loss_names = ("loss", "box_loss", "cls_loss", "l1_loss")
        row = {"epoch": epoch + 1}
        row.update({f"train/{name}": train_results[name] for name in loss_names})
        if teacher is not None:
            row["train/distill_loss"] = train_results["distill_loss"]
        row.update({
            "metrics/precision": val_results["precision"],
            "metrics/recall": val_results["recall"],
            "metrics/mAP50": val_results["map50"],
            "metrics/mAP50-95": val_results["map50_95"],
        })
        if teacher is not None:

            row["metrics/agnostic_mAP50"] = val_results["agnostic_map50"]
        row.update({f"val/{name}": val_results[name] for name in loss_names})
        row["lr"] = epoch_lr
        log.log_epoch_results(row, checkpoint_path.parent)

    if not checkpoint_path.is_file():
        message = "No best checkpoint was saved during this run."
        logger.error(message)
        raise RuntimeError(message)
    return checkpoint_path

# Settings that may change when resuming: they affect speed and logging, not the training itself
RESUME_OVERRIDES = ("num_workers", "console_level")


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset-path", type=Path, default=None,
        help="DSEC root with labelled train and val directories. Required unless resuming.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, metavar="RUN_DIR",
        help="Continue an interrupted run (e.g. runs/student/exp_006) from its latest checkpoint. "
             "All settings come from the original run; only --num-workers and --console-level "
             "may be given again.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1, help="Total number of epochs to train.")
    parser.add_argument(
        "--degradation-augmentation", action="store_true",
        help="Randomly mix clean, dark and overexposed images for RGB-only student training.",
    )
    parser.add_argument(
        "--augmentation-seed", type=int, default=DEFAULT_AUGMENTATION_SEED,
        help="Non-negative seed for training augmentation only (default: 0); independent of evaluation.",
    )
    parser.add_argument(
        "--degradation-shares", type=float, nargs=3,
        metavar=("CLEAN", "DARK", "OVEREXPOSED"), default=DEFAULT_DEGRADATION_SHARES,
        help="Condition probabilities summing to 1 (default: 0.5 0.25 0.25).",
    )
    parser.add_argument(
        "--dark-strength-range", type=float, nargs=2, metavar=("MIN", "MAX"),
        default=DEFAULT_STRENGTH_RANGE,
        help="Uniform training darkening strength range in [0, 1] (default: 0.35 1.0).",
    )
    parser.add_argument(
        "--overexposed-strength-range", type=float, nargs=2, metavar=("MIN", "MAX"),
        default=DEFAULT_STRENGTH_RANGE,
        help="Uniform training over-exposure strength range in [0, 1] (default: 0.35 1.0).",
    )
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Run training and validation only; do not run final eval step.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit each of train/val/test to this many evenly spaced frames (for quick smoke runs).",
    )
    parser.add_argument(
        "--teacher-checkpoint", type=Path, default=None,
        help="Frozen RGB + event teacher to distil from. Training then uses paired RGB + event "
             "batches, while the student, validation and test stay RGB-only.",
    )
    parser.add_argument(
        "--lambda-distill", type=float, default=1.0,
        help="Weight of the distillation term added to the detection loss.",
    )
    parser.add_argument(
        "--distill-mode", choices=DISTILL_MODES, default="unweighted",
        help="Where the distillation term compares features: 'unweighted' everywhere (baseline) "
             "or 'object' only inside the enlarged labelled boxes. Needs --teacher-checkpoint.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for model initialisation, shuffling and augmentation. Use the same seed for "
             "every student you compare.",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Train on unaugmented frames, e.g. for the tiny-subset overfit test.",
    )
    parser.add_argument(
        "--console-level", type = str, default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help = "Set the logging level.",
    )
    args = parser.parse_args()
    if args.resume is not None:
        args = resumed_args(parser, args)
    elif args.dataset_path is None:
        parser.error("--dataset-path is required unless resuming")
    try:
        args.augmentation_seed = validate_augmentation_seed(args.augmentation_seed)
        args.degradation_shares = validate_degradation_shares(args.degradation_shares)
        args.dark_strength_range = validate_strength_range(args.dark_strength_range)
        args.overexposed_strength_range = validate_strength_range(args.overexposed_strength_range)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.lambda_distill < 0:
        parser.error("--lambda-distill must be non-negative")
    if args.distill_mode != "unweighted" and args.teacher_checkpoint is None:
        parser.error("--distill-mode needs --teacher-checkpoint")
    return args


def resumed_args(parser: argparse.ArgumentParser, given: argparse.Namespace) -> argparse.Namespace:
    """Rebuild a run's original settings from its folder, keeping the allowed overrides."""
    run_directory = given.resume
    if not run_directory.is_dir():
        parser.error(f"--resume: run folder not found: {run_directory}")

    changed = [
        "--" + name.replace("_", "-")
        for name, value in vars(given).items()
        if name not in ("resume", *RESUME_OVERRIDES) and value != parser.get_default(name)
    ]
    if changed:
        parser.error(
            f"--resume takes its settings from the original run, so {', '.join(changed)} cannot "
            "be changed. Start a new run instead."
        )

    original_command = log.load_command(run_directory)
    script = Path(sys.argv[0]).name
    if Path(original_command[0]).name != script:
        parser.error(
            f"--resume: {run_directory} was started with {Path(original_command[0]).name}, "
            f"not {script}"
        )

    args = parser.parse_args(original_command[1:])
    args.resume = run_directory
    for name in RESUME_OVERRIDES:
        if getattr(given, name) != parser.get_default(name):
            setattr(args, name, getattr(given, name))
    return args

def main(role: str, rgb_only: bool) -> None:
    """Train, validate and test one model. `role` is "student", "teacher" or "distill"."""
    input_channels = 3 if rgb_only else 5
    description = (
        "Train and validate the RGB student model." if rgb_only
        else "Train and validate the RGB + event teacher model."
    )
    args = parse_args(description)
    if args.degradation_augmentation and (not rgb_only or args.teacher_checkpoint is not None):
        raise ValueError("--degradation-augmentation only applies to RGB-only training without a teacher")
    resuming = args.resume is not None

    log_directory = args.resume if resuming else log.create_log_directory(role)
    latest_path = log_directory / f"latest_{role}_yolo.pt"

    logger = log.create_logger(
        "train",
        log_directory,
        console_level=args.console_level,
    )
    if resuming:
        if not latest_path.is_file():
            message = f"Cannot resume: no latest checkpoint at {latest_path}. Start a new run instead."
            logger.error(message)
            raise FileNotFoundError(message)
        logger.info("Resuming %s with: %s", log_directory, " ".join(sys.argv))
    else:
        log.log_command(sys.argv, log_directory)

    device = select_device()
    # Seed before anything random is built: the student's fresh head, the shuffle, augmentation
    init_seeds(args.seed, deterministic=True)

    distilling = args.teacher_checkpoint is not None
    if distilling and not rgb_only:
        message = "--teacher-checkpoint only applies to the RGB student."
        logger.error(message)
        raise ValueError(message)

    model = extract_trainable_model(
        ch=input_channels,
        weights_path="yolo26n.pt",
        device=device,
        epochs=args.epochs,
    )
    # Build the teacher after the student: its random initialisation (overwritten by the checkpoint)
    # would otherwise shift the student's fresh head away from the RGB baseline's for the same seed
    teacher = load_teacher(args.teacher_checkpoint, device, logger) if distilling else None

    loader_settings = dict(
        dataset_path = args.dataset_path,
        logger = logger,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        max_samples = args.max_samples,
    )
    augmentation_settings = (
        {
            "augmentation_shares": args.degradation_shares,
            "dark_strength_range": args.dark_strength_range,
            "overexposed_strength_range": args.overexposed_strength_range,
            "augmentation_seed": args.augmentation_seed,
        } if args.degradation_augmentation else {}
    )
    train_settings = dict(augment=not args.no_augment, seed=args.seed)
    train_loader, val_loader = create_dataloaders(
        rgb_only=rgb_only, **loader_settings, **train_settings, **augmentation_settings,
    )
    if distilling:
        # The teacher needs the paired event channels, so training re-reads with events while
        # validation keeps the RGB-only loader built above. Augmentation makes the same random
        # draws either way, so the student sees the same RGB views as the RGB-only baseline.
        train_loader, _ = create_dataloaders(rgb_only=False, **loader_settings, **train_settings)

    optimiser = create_optimiser(model, batch_size=args.batch_size)

    config = {
        'Model': role,
        'Input Channels': input_channels,
        'Selected Device': str(device),
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
        'Seed': args.seed,
        'Flip Scale Colour Augmentation': not args.no_augment,
        'Nominal Batch Size': NOMINAL_BATCH_SIZE,
        'Warmup Epochs': WARMUP_EPOCHS,
        'Optimiser': type(optimiser).__name__,
        'Learning Rate': OPTIMISER_LR,
        'Deterministic': True,
        'Teacher Checkpoint': str(args.teacher_checkpoint) if distilling else None,
        'Lambda Distill': args.lambda_distill if distilling else None,
        'Distill Mode': args.distill_mode if distilling else None,
        'Teacher Confidence Threshold': (
            TEACHER_CONFIDENCE_THRESHOLD if distilling else None
        ),
        'Object Mask Enlargement': MASK_ENLARGEMENT if distilling and args.distill_mode == "object" else None,
        'Training Augmentation': {
            'enabled': args.degradation_augmentation,
            'shares': dict(zip(('clean', 'dark', 'overexposed'), args.degradation_shares)),
            'dark_strength_range': list(args.dark_strength_range),
            'overexposed_strength_range': list(args.overexposed_strength_range),
            'seed': args.augmentation_seed,
        },
    }
    if args.max_samples is not None:
        config['Training Sequences'] = sequence_names(train_loader.dataset)
        config['Validation Sequences'] = sequence_names(val_loader.dataset)
    log.log_model_config(config, log_directory, logger)

    best_path = log_directory / f"best_{role}_yolo.pt"

    checkpoint_path = train_model(
        train_loader,
        val_loader,
        model,
        optimiser,
        device,
        logger,
        total_epochs=args.epochs,
        checkpoint_path=best_path,
        latest_path=latest_path,
        teacher=teacher,
        lambda_distill=args.lambda_distill,
        distill_mode=args.distill_mode,
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
