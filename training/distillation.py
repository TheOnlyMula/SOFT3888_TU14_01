"""Feature distillation from a frozen privileged teacher to the RGB student.

The teacher sees RGB + event channels; the student only ever sees the RGB channels of the
same frame. feature_distillation_loss is the basic (unweighted) baseline: every spatial location
counts equally. object_region_mask marks the labelled object regions that the object-aware
variant restricts the loss to, and quality_weight gives each image the deterministic weight
w = clip(s_e * c_t, 0, 1) that the quality-weighted variant scales that loss by. This module is
training-only: inference never imports it.
"""
import logging, json
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.model import build_model, load_weights

# Neck features fed to the detection head. The student and teacher share this architecture,
# so the feature maps already match and need no projection layer. Both head branches hold the
# same neck features, but in training mode the one2one branch gets them detached, so only the
# one2many copy lets the distillation gradient reach the student's backbone and neck.
DISTILL_TARGET = "one2many"

#Level 0  is P3 (Stride 8). Levels 1 and 2 are P4 (stride 16) and P5 (stride 32).
DISTILL_LEVEL = 0

MASK_ENLARGEMENT = 1.2

TEACHER_CONFIDENCE_THRESHOLD = 0.25

# Event coverage (loaders.dataset_loader.event_coverage) at which a window counts as fully dense: the median
# over every training frame, so only windows sparser than the typical frame get s_e < 1. Measured
# by tools/measure_event_density.py and recorded in docs/event_density.json.
# Originally, the event density reference is a hard-coded value that was obtained as above for the full DSEC
# dataset (0.4353). If docs/event_density.json exists, the event density reference will be the value recorded in the file.
DEFAULT_EVENT_DENSITY_REFERENCE = 0.4353
def get_event_density_reference() -> float:
    try:
        with open(Path(__file__).resolve().parents[1] / "docs" / "event_density.json") as f:
            density_json = json.load(f)
            return density_json["event_density_reference"]
    except (FileNotFoundError, KeyError) as e:
        return DEFAULT_EVENT_DENSITY_REFERENCE
EVENT_DENSITY_REFERENCE = get_event_density_reference()

DISTILL_MODES = ("unweighted", "object", "object-quality")

# Keeps the object-aware loss finite (zero) for images without labelled objects.
MASK_EPS = 1e-6


def event_density(coverage: torch.Tensor) -> torch.Tensor:
    """s_e for each image: its event window's coverage, normalised so EVENT_DENSITY_REFERENCE is 1.

    `coverage` is measured by the dataset on the raw event window, before augmentation warps it.
    """
    return (coverage / EVENT_DENSITY_REFERENCE).clamp(0, 1)


def load_teacher(
    checkpoint_path: Path,
    device: torch.device,
    logger: logging.Logger,
    ch: int = 5,
) -> torch.nn.Module:
    """Load the privileged teacher, frozen and in eval mode."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        message = f"Teacher checkpoint not found: {checkpoint_path}"
        logger.error(message)
        raise FileNotFoundError(message)

    teacher = load_weights(build_model(ch), checkpoint_path, device)
    teacher = teacher.to(device).eval().requires_grad_(False)
    # Output final one-to-one detections for teacher_confidence, whatever Ultralytics' default
    teacher.end2end = True

    logger.info("Loaded frozen teacher: %s (input channels = %d)", checkpoint_path, ch)
    return teacher


def head_outputs(preds) -> dict:
    """The head's prediction dict, whether the model ran in train or eval mode."""
    return preds[1] if isinstance(preds, (list, tuple)) else preds


def teacher_confidence(teacher_preds, threshold: float = TEACHER_CONFIDENCE_THRESHOLD) -> torch.Tensor:
    detections = teacher_preds[0] if isinstance(teacher_preds, (list, tuple)) else teacher_preds
    if detections.shape[-1] != 6:
        raise ValueError("teacher_confidence needs end-to-end B x N x 6 detections")
    confidence = detections[..., 4]
    retained = confidence >= threshold
    return (confidence * retained).sum(dim=1) / retained.sum(dim=1).clamp(min=1)


def quality_weight(event_densities: torch.Tensor, teacher_confidences: torch.Tensor) -> torch.Tensor:
    """The deterministic quality weight w = clip(s_e * c_t, 0, 1), one value per image.

    s_e is event_density of the image's event window and c_t is teacher_confidence of the frozen
    teacher's output, so a sparse event window or an unsure teacher lowers w. Neither carries a
    gradient, so nothing in w is learned.
    """
    return (event_densities * teacher_confidences).clamp(0, 1)


def object_region_mask(
    batch_idx: torch.Tensor,
    bboxes: torch.Tensor,
    batch_size: int,
    feature_size: tuple[int, int],
    enlarge: float = MASK_ENLARGEMENT,
) -> torch.Tensor:
    height, width = feature_size
    mask = torch.zeros(batch_size, height, width, device=bboxes.device)
    if len(bboxes) == 0:
        return mask

    centre_x, centre_y, box_w, box_h = bboxes.unbind(1)
    half_w = box_w * enlarge / 2
    half_h = box_h * enlarge / 2
    x1 = ((centre_x - half_w) * width).clamp(0, width)
    x2 = ((centre_x + half_w) * width).clamp(0, width)
    y1 = ((centre_y - half_h) * height).clamp(0, height)
    y2 = ((centre_y + half_h) * height).clamp(0, height)

    cols = torch.arange(width, device=bboxes.device)
    rows = torch.arange(height, device=bboxes.device)
    in_x = (cols + 1 > x1[:, None]) & (cols < x2[:, None])
    in_y = (rows + 1 > y1[:, None]) & (rows < y2[:, None])
    box_masks = in_y[:, :, None] & in_x[:, None, :]

    for image in range(batch_size):
        mine = batch_idx == image
        if mine.any():
            mask[image] = box_masks[mine].any(dim=0).float()
    return mask


def feature_distillation_loss(student_preds, teacher_preds, level: int = DISTILL_LEVEL) -> torch.Tensor:
    """Smooth L1 distance between the student's and teacher's features at one neck level.

    The teacher side is detached, so no gradient reaches the frozen teacher. Every spatial
    location counts equally: this is the unweighted baseline.
    """
    student_feature, teacher_feature = distilled_features(student_preds, teacher_preds, level)
    return F.smooth_l1_loss(student_feature, teacher_feature.detach())


def object_distillation_loss(
    student_preds, teacher_preds, mask: torch.Tensor, level: int = DISTILL_LEVEL
) -> torch.Tensor:
    """Distillation restricted to the object mask.

    The distance at each feature cell is the smooth L1 averaged over channels, so with an all-ones
    mask the batch mean equals feature_distillation_loss and one lambda_distill serves both modes.
    Each image's loss averages over its masked cells; an image without objects contributes 0.
    """
    student_feature, teacher_feature = distilled_features(student_preds, teacher_preds, level)
    distance = F.smooth_l1_loss(student_feature, teacher_feature.detach(), reduction="none").mean(dim=1)
    return (mask * distance).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + MASK_EPS)


def distilled_features(student_preds, teacher_preds, level: int = DISTILL_LEVEL):
    """The student's and teacher's neck features at the distilled level."""
    return (
        head_outputs(student_preds)[DISTILL_TARGET]["feats"][level],
        head_outputs(teacher_preds)[DISTILL_TARGET]["feats"][level],
    )
