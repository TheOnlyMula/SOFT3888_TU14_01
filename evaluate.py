import argparse
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO


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


def validate_model(
    val_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[float, float, float, float]:
    """Evaluate validation loss without updating model weights."""
    if len(val_loader) == 0:
        message = "Validation loader must be non-empty."
        logger.error(message)
        raise ValueError(message)

    model.eval()
    total_loss = 0.0
    total_box_loss = 0.0
    total_cls_loss = 0.0
    total_l1_loss = 0.0
    logger.info("Starting validation")

    with torch.no_grad():
        for batch_index, (imgs, targets) in enumerate(val_loader):
            loss, loss_items = model(_prepare_yolo_batch(imgs, targets, device))

            step_loss = loss.sum().item()
            step_box_loss = float(loss_items["box_loss"])
            step_cls_loss = float(loss_items["cls_loss"])
            step_l1_loss = float(loss_items["l1_loss"])

            total_loss += step_loss
            total_box_loss += step_box_loss
            total_cls_loss += step_cls_loss
            total_l1_loss += step_l1_loss

            if batch_index > 0 and batch_index % 10 == 0:
                logger.debug(
                    "Validation batch completed step = %d/%d "
                    "loss = %.4f box_loss = %.4f cls_loss = %.4f l1_loss = %.4f",
                    batch_index,
                    len(val_loader),
                    step_loss,
                    step_box_loss,
                    step_cls_loss,
                    step_l1_loss,
                )

    num_batches = len(val_loader)
    return (
        total_loss / num_batches,
        total_box_loss / num_batches,
        total_cls_loss / num_batches,
        total_l1_loss / num_batches,
    )


def evaluate_test_model(
    test_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    logger: logging.Logger,
    checkpoint_path: Path,
) -> tuple[float, float]:
    """Restore the best checkpoint and report average test loss."""
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

    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_images = 0
    logger.info("Loaded best checkpoint: %s", checkpoint_path)

    with torch.no_grad():
        for imgs, targets in test_loader:
            loss, _ = model(_prepare_yolo_batch(imgs, targets, device))
            batch_loss = loss.sum().item()
            if not math.isfinite(batch_loss):
                message = "Non-finite test loss; no valid test summary is available."
                logger.error(message)
                raise ValueError(message)

            total_loss += batch_loss
            total_batches += 1
            total_images += imgs.size(0)

    if total_batches == 0 or total_images == 0:
        message = "No test images were evaluated."
        logger.error(message)
        raise ValueError(message)

    avg_test_batch_loss = total_loss / total_batches
    avg_test_image_loss = total_loss / total_images
    logger.info("Test samples: %d | Test batches: %d", total_images, total_batches)
    logger.info("Average test batch loss: %.4f", avg_test_batch_loss)
    logger.info("Average test per-image loss: %.4f", avg_test_image_loss)
    return avg_test_batch_loss, avg_test_image_loss


def evaluate_map_model(
    checkpoint_path: Path,
    model_yaml: Path | str,
    data_yaml: Path | str,
    device: torch.device,
    logger: logging.Logger,
):
    """Run Ultralytics mAP evaluation on the configured test split."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        message = f"Best checkpoint not found: {checkpoint_path}"
        logger.error(message)
        raise FileNotFoundError(message)

    logger.info("Loading best weights for mAP evaluation: %s", checkpoint_path)
    yolo_model = YOLO(str(model_yaml), task="detect")
    yolo_model.model.names = {
        0: "pedestrian",
        1: "rider",
        2: "car",
        3: "bus",
        4: "truck",
        5: "bicycle",
        6: "motorcycle",
        7: "train",
    }

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    yolo_model.model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Starting mAP evaluation on the test split")
    metrics = yolo_model.val(
        data=str(data_yaml),
        split="test",
        device=str(device),
        plots=True,
    )
    logger.info("Test mAP@50: %.4f", metrics.box.map50)
    logger.info("Test mAP@50-95: %.4f", metrics.box.map)
    return metrics


if __name__ == "__main__":
    import log
    import sys

    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained weights (.pt)",
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["student", "teacher"],
        help="Evaluating student or teacher",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_yaml = script_dir / "configs" / "dsec_rgb.yaml"
    model_yaml = script_dir / "configs" / f"yolo26n_{args.type}.yaml"

    log_directory = log.create_log_directory(args.type)
    logger = log.create_logger("evaluate", log_directory)
    log.log_command(sys.argv, log_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_map_model(args.weights, model_yaml, data_yaml, device, logger)
