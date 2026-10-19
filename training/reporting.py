"""What the training loop reports after each epoch: train.log lines and the results.csv row."""
import logging

import torch

LOSS_NAMES = ("loss", "box_loss", "cls_loss", "l1_loss")


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes}m{seconds:02d}s"


def quality_weight_stats(quality_weights: list[torch.Tensor]) -> dict:
    """Mean, minimum and maximum w over an epoch's training images, and the fraction with w = 0.

    Empty when no weights were collected, i.e. outside "object-quality" mode.
    """
    if not quality_weights:
        return {}
    weights = torch.cat(quality_weights)
    return {
        "mean": weights.mean().item(),
        "min": weights.min().item(),
        "max": weights.max().item(),
        "zero_fraction": (weights == 0).float().mean().item(),
    }


def log_epoch_summary(
    logger: logging.Logger,
    train_results: dict,
    val_results: dict,
    weight_stats: dict,
    distill_mode: str,
    lambda_distill: float,
) -> None:
    """Log an epoch's training and validation losses, distillation terms and validation metrics."""
    for split, results in (("train", train_results), ("val", val_results)):
        logger.info(
            "  %-5s loss = %.4f box_loss = %.4f cls_loss = %.4f l1_loss = %.4f",
            split,
            results["loss"],
            results["box_loss"],
            results["cls_loss"],
            results["l1_loss"],
        )
    # Only runs with a teacher have a distillation loss
    if "distill_loss" in train_results:
        logger.info(
            "  train distill_loss = %.4f (mode = %s, lambda = %.3g, contribution = %.4f)",
            train_results["distill_loss"],
            distill_mode,
            lambda_distill,
            lambda_distill * train_results["distill_loss"],
        )
    if weight_stats:
        logger.info(
            "  train quality_weight mean = %.4f min = %.4f max = %.4f zero_fraction = %.4f",
            weight_stats["mean"],
            weight_stats["min"],
            weight_stats["max"],
            weight_stats["zero_fraction"],
        )
    logger.info(
        "  val   precision = %.4f recall = %.4f mAP50 = %.4f mAP50-95 = %.4f agnostic_mAP50 = %.4f",
        val_results["precision"],
        val_results["recall"],
        val_results["map50"],
        val_results["map50_95"],
        val_results["agnostic_map50"],
    )


def results_row(
    epoch: int, train_results: dict, val_results: dict, weight_stats: dict, lr: float
) -> dict:
    """One epoch's row of results.csv. `epoch` is 1-based; column order is the file's order."""
    # Only runs with a teacher have a distillation loss
    distilling = "distill_loss" in train_results
    row = {"epoch": epoch}
    row.update({f"train/{name}": train_results[name] for name in LOSS_NAMES})
    if distilling:
        row["train/distill_loss"] = train_results["distill_loss"]
    # No "loss" or "metric" in these names, so plot_results leaves them out of its plot grid
    row.update({f"train/quality_weight_{name}": value for name, value in weight_stats.items()})
    row.update({
        "metrics/precision": val_results["precision"],
        "metrics/recall": val_results["recall"],
        "metrics/mAP50": val_results["map50"],
        "metrics/mAP50-95": val_results["map50_95"],
    })
    if distilling:
        row["metrics/agnostic_mAP50"] = val_results["agnostic_map50"]
    row.update({f"val/{name}": val_results[name] for name in LOSS_NAMES})
    row["lr"] = lr
    return row
