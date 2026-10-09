import argparse
import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics
from dataset_loader import (
    create_test_loader,
    resolve_degradation_strength,
    rgb_loader_fingerprint,
)
from overexposure import STRENGTH_LEVELS
from utils import extract_trainable_model, select_device


def _prepare_yolo_batch(imgs, targets, device):
    imgs = imgs.to(device)
    targets = targets.to(device)
    B, _, H, W = imgs.shape
    return {
        "img": imgs,
        "cls": targets[:, 1:2],
        "bboxes": targets[:, 2:],
        "batch_idx": targets[:, 0],
        "im_file": [f"image_{i}.png" for i in range(B)],
        "ori_shape": [(H, W)] * B,
        "ratio_pad": [((1.0, 1.0), (0.0, 0.0))] * B,
    }


class DsecDetectionValidator(DetectionValidator):
    """Ultralytics detection validator"""

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.agnostic_metrics = DetMetrics()
        self.agnostic_metrics.names = {0: "object"}

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict) -> None:
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            pbatch["cls"] = torch.zeros_like(pbatch["cls"])
            predn = {**pred, "cls": torch.zeros_like(pred["cls"])}
            target_cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.agnostic_metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": target_cls,
                    "target_img": np.unique(target_cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    "im_name": pbatch["im_file"],
                }
            )

    def get_stats(self) -> dict:
        stats = super().get_stats()
        self.agnostic_metrics.process()
        return stats


def per_class_results(validator: DsecDetectionValidator) -> list[dict]:
    metrics = validator.metrics
    overall = summarise_detection_stats(validator)
    rows = [
        {
            "class": "all",
            "images": int(validator.seen),
            "instances": int(metrics.nt_per_class.sum()),
            "precision": overall["precision"],
            "recall": overall["recall"],
            "map50": overall["map50"],
            "map50_95": overall["map50_95"],
        }
    ]
    for i, class_index in enumerate(metrics.ap_class_index):
        precision, recall, map50, map50_95 = metrics.class_result(i)
        rows.append(
            {
                "class": validator.names[int(class_index)],
                "images": int(metrics.nt_per_image[class_index]),
                "instances": int(metrics.nt_per_class[class_index]),
                "precision": float(precision),
                "recall": float(recall),
                "map50": float(map50),
                "map50_95": float(map50_95),
            }
        )
    return rows


def format_results_table(rows: list[dict]) -> list[str]:
    """Lines of a results table"""
    header = ("%22s" + "%11s" * 6) % (
        "Class",
        "Images",
        "Instances",
        "Box(P",
        "R",
        "mAP50",
        "mAP50-95)",
    )
    row_format = "%22s" + "%11i" * 2 + "%11.3g" * 4
    lines = [header]
    for row in rows:
        lines.append(
            row_format
            % (
                row["class"],
                row["images"],
                row["instances"],
                row["precision"],
                row["recall"],
                row["map50"],
                row["map50_95"],
            )
        )
    return lines


def summarise_detection_stats(validator: DsecDetectionValidator) -> dict[str, float]:
    box = validator.metrics.box
    return {
        "precision": float(box.mp),  # mean precision over classes
        "recall": float(box.mr),  # mean recall over classes
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "agnostic_map50": float(validator.agnostic_metrics.box.map50),
    }


def create_metrics_validator(
    model: torch.nn.Module, device: torch.device, save_dir: Path | None = None
) -> DsecDetectionValidator:
    plots = save_dir is not None
    if plots:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    validator = DsecDetectionValidator(
        save_dir=Path(save_dir) if plots else Path(tempfile.gettempdir()),
        args={"plots": plots},
    )
    validator.device = device
    validator.data = {}
    validator.training = False
    validator.init_metrics(model)
    return validator


def validate_model(
    val_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    logger: logging.Logger,
    compute_metrics: bool = True,
    save_dir: Path | None = None,
) -> dict:
    """Evaluate loss and detection metrics in one pass, without updating model weights."""
    if len(val_loader) == 0:
        message = "Validation loader must be non-empty."
        logger.error(message)
        raise ValueError(message)

    model.eval()
    totals = {"loss": 0.0, "box_loss": 0.0, "cls_loss": 0.0, "l1_loss": 0.0}
    validator = (
        create_metrics_validator(model, device, save_dir) if compute_metrics else None
    )
    logger.info("Starting validation")

    with torch.no_grad():
        for batch_index, (imgs, targets) in enumerate(val_loader):
            batch = _prepare_yolo_batch(imgs, targets, device)
            preds = model(batch["img"])
            loss, loss_items = model.loss(batch, preds)

            step = {"loss": loss.sum().item() / imgs.shape[0]}
            step.update(
                {
                    name: float(loss_items[name])
                    for name in ("box_loss", "cls_loss", "l1_loss")
                }
            )
            for name, value in step.items():
                totals[name] += value

            if validator is not None:
                detections = validator.postprocess(preds)
                validator.update_metrics(detections, batch)
                if save_dir is not None and batch_index < 3:
                    # Plot the RGB channels only, so teacher inputs with event channels plot too.
                    rgb_batch = {**batch, "img": batch["img"][:, :3]}
                    validator.plot_val_samples(rgb_batch, batch_index)
                    validator.plot_predictions(rgb_batch, detections, batch_index)

            if batch_index > 0 and batch_index % 10 == 0:
                logger.debug(
                    "Validation batch completed step = %d/%d "
                    "loss = %.4f box_loss = %.4f cls_loss = %.4f l1_loss = %.4f",
                    batch_index,
                    len(val_loader),
                    step["loss"],
                    step["box_loss"],
                    step["cls_loss"],
                    step["l1_loss"],
                )

    results = {name: total / len(val_loader) for name, total in totals.items()}
    if validator is not None:
        validator.get_stats()
        results.update(summarise_detection_stats(validator))
        results["per_class"] = per_class_results(validator)
        if save_dir is not None:
            validator.finalize_metrics()
    return results


def evaluate_test_model(
    test_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    logger: logging.Logger,
    checkpoint_path: Path,
    save_dir: Path | None = None,
    compute_metrics: bool = True,
) -> dict:
    """Restore the best checkpoint and evaluate it on the test split.

    Uses the same evaluation as per-epoch validation, so test and validation numbers are directly
    comparable. Returns loss, detection metrics and per-class results.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        message = f"Best checkpoint not found: {checkpoint_path}"
        logger.error(message)
        raise FileNotFoundError(message)

    if len(test_loader) == 0:
        message = "The test loader is empty; no test loss can be calculated."
        logger.error(message)
        raise ValueError(message)

    model.to(device)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        message = f"Could not load the best checkpoint: {checkpoint_path}"
        logger.error(message)
        raise RuntimeError(message) from exc

    logger.info("Loaded best checkpoint: %s", checkpoint_path)
    results = validate_model(
        test_loader,
        model,
        device,
        logger,
        compute_metrics=compute_metrics,
        save_dir=save_dir,
    )

    if not math.isfinite(results["loss"]):
        message = "Non-finite test loss; no valid test summary is available."
        logger.error(message)
        raise ValueError(message)

    logger.info(
        "Test samples: %d | Test batches: %d",
        len(test_loader.dataset),
        len(test_loader),
    )
    logger.info(
        "Test loss = %.4f box_loss = %.4f cls_loss = %.4f l1_loss = %.4f",
        results["loss"],
        results["box_loss"],
        results["cls_loss"],
        results["l1_loss"],
    )
    if compute_metrics:
        for line in format_results_table(results["per_class"]):
            logger.info(line)
        logger.info(
            "Class-agnostic mAP50 = %.3g (all classes merged into one)",
            results["agnostic_map50"],
        )
    if save_dir is not None:
        logger.info("Test plots saved to %s", save_dir)
    return results


def degradation_strength_argument(value: str) -> float:
    """Accept a named level or a number on the command line."""
    try:
        return resolve_degradation_strength(value if value in STRENGTH_LEVELS else float(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def evaluate_degraded_benchmark(
    dataset_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    logger: logging.Logger,
    checkpoint_path: Path,
    rgb_only: bool,
    batch_size: int = 16,
    num_workers: int = 8,
    max_samples: int | None = None,
    degradation_strength: str | float = 1.0,
    evaluation_seed: int = 0,
) -> dict[str, dict]:
    conditions = ("clean", "dark", "overexposed")
    results = {}

    for condition in conditions:
        loader = create_test_loader(
            dataset_path,
            logger,
            rgb_only=rgb_only,
            batch_size=batch_size,
            num_workers=num_workers,
            max_samples=max_samples,
            degradation=condition,
            degradation_strength=degradation_strength,
            evaluation_seed=evaluation_seed,
        )

        fingerprint = rgb_loader_fingerprint(loader)

        condition_results = evaluate_test_model(
            loader,
            model,
            device,
            logger,
            checkpoint_path,
        )

        results[condition] = {
            "map50": condition_results["map50"],
            "map50_95": condition_results["map50_95"],
            "recall": condition_results["recall"],
            "fingerprint": fingerprint,
        }

    clean = results["clean"]

    for condition in ("dark", "overexposed"):
        results[condition]["map50_drop"] = clean["map50"] - results[condition]["map50"]
        results[condition]["map50_95_drop"] = (
            clean["map50_95"] - results[condition]["map50_95"]
        )
        results[condition]["recall_drop"] = (
            clean["recall"] - results[condition]["recall"]
        )

    results["clean"]["map50_drop"] = 0.0
    results["clean"]["map50_95_drop"] = 0.0
    results["clean"]["recall_drop"] = 0.0

    return results


def format_benchmark_table(results: dict[str, dict]) -> list[str]:
    header = (
        f"{'Condition':<15}"
        f"{'mAP50':>10}"
        f"{'mAP50-95':>12}"
        f"{'Recall':>10}"
        f"{'mAP50 Drop':>12}"
        f"{'mAP50-95 Drop':>15}"
        f"{'Recall Drop':>13}"
    )

    lines = [header]

    for condition in ("clean", "dark", "overexposed"):
        row = results[condition]
        lines.append(
            f"{condition:<15}"
            f"{row['map50']:>10.3f}"
            f"{row['map50_95']:>12.3f}"
            f"{row['recall']:>10.3f}"
            f"{row['map50_drop']:>12.3f}"
            f"{row['map50_95_drop']:>15.3f}"
            f"{row['recall_drop']:>13.3f}"
        )

    return lines


if __name__ == "__main__":
    import log
    import sys

    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to the trained weights (.pt)"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["student", "teacher"],
        help="Evaluating student or teacher",
    )
    parser.add_argument(
        "--dataset-path", type=Path, required=True, help="Path to the dataset directory"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Evaluate clean, dark and overexposed benchmark conditions.",
    )
    parser.add_argument(
        "--degradation-strength",
        type=degradation_strength_argument,
        default=1.0,
        help='Degradation strength: "mild", "strong", or a number in [0, 1].',
    )
    parser.add_argument("--evaluation-seed", type=int, default=0)
    args = parser.parse_args()

    log_directory = log.create_log_directory(args.type)
    logger = log.create_logger("evaluate", log_directory)
    log.log_command(sys.argv, log_directory)
    device = select_device()

    rgb_only = args.type == "student"
    model = extract_trainable_model(
        ch=3 if rgb_only else 5,
        weights_path=None,
        device=device,
    )

    if args.benchmark:
        benchmark_results = evaluate_degraded_benchmark(
            dataset_path=args.dataset_path,
            model=model,
            device=device,
            logger=logger,
            checkpoint_path=args.weights,
            rgb_only=rgb_only,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            degradation_strength=args.degradation_strength,
            evaluation_seed=args.evaluation_seed,
        )

        for line in format_benchmark_table(benchmark_results):
            logger.info(line)

        for condition, result in benchmark_results.items():
            logger.info(
                "%s fingerprint: %s",
                condition,
                result["fingerprint"],
            )

    else:
        test_loader = create_test_loader(
            args.dataset_path,
            logger,
            rgb_only=rgb_only,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )

        evaluate_test_model(
            test_loader,
            model,
            device,
            logger,
            args.weights,
            save_dir=log_directory / "test",
        )
