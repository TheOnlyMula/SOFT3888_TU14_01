"""Feature distillation from a frozen privileged teacher to the RGB student.

The teacher sees RGB + event channels; the student only ever sees the RGB channels of the
same frame. feature_distillation_loss is the basic (unweighted) baseline: every spatial location
counts equally. object_region_mask marks the labelled object regions that the object-aware
variant restricts the loss to. This module is training-only: inference never imports it.
"""
import logging
import math
import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from utils import build_model, load_weights

# Neck features fed to the detection head. The student and teacher share this architecture,
# so the feature maps already match and need no projection layer. Both head branches hold the
# same neck features, but in training mode the one2one branch gets them detached, so only the
# one2many copy lets the distillation gradient reach the student's backbone and neck.
DISTILL_TARGET = "one2many"

#Level 0  is P3 (Stride 8). Levels 1 and 2 are P4 (stride 16) and P5 (stride 32).
DISTILL_LEVEL = 0

MASK_ENLARGEMENT = 1.2

DISTILL_MODES = ("unweighted", "object")

# Keeps the object-aware loss finite (zero) for images without labelled objects.
MASK_EPS = 1e-6


@lru_cache(maxsize=1)
def load_event_density_reference() -> float:
    """Read the frozen training-only coverage reference once per process."""
    path = Path(__file__).resolve().parent / "docs" / "event_density_reference.json"
    with path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if (metadata["split"] != "train" or metadata["region"] != "whole_frame"
            or metadata["event_window_us"] != 50_000):
        raise ValueError("event-density reference must use training-only causal 50 ms whole-frame coverage")
    reference = float(metadata["event_density_reference"])
    if not math.isfinite(reference) or not 0 < reference <= 1:
        raise ValueError("event-density reference must be finite and in (0, 1]")
    return reference


def event_density(teacher_images: torch.Tensor, reference: float | None = None) -> torch.Tensor:
    """Return one deterministic whole-frame event-density score per teacher input.

    ``teacher_images`` is the same B x 5 x H x W RGB + event tensor passed to
    the teacher. Its last two channels hold the causal 50 ms histogram built
    by DSECFusionDataset. Log scaling and clipping preserve event occupancy.
    A pixel counts once if either polarity has at least one event, regardless
    of event count. The score is active pixels / (H * W) / reference, clipped
    to [0, 1]. No resizing, resampling or separate event-window read is used.

    By default, ``reference`` is loaded from docs/event_density_reference.json.
    An explicit reference must be a fixed positive coverage fraction shared
    by all frames, rather than recomputed for each frame or batch.
    This helper computes the score only; it does not change the training loss.
    """
    if reference is None:
        reference = load_event_density_reference()
    if not math.isfinite(reference) or not 0 < reference <= 1:
        raise ValueError("event-density reference must be finite and in (0, 1]")
    if teacher_images.ndim != 4 or teacher_images.shape[1] != 5:
        raise ValueError("teacher images must have shape (B, 5, H, W)")
    if teacher_images.shape[2] == 0 or teacher_images.shape[3] == 0:
        raise ValueError("teacher images must have non-empty spatial dimensions")

    active = (teacher_images[:, 3:5].detach() > 0).any(dim=1)
    coverage = active.to(dtype=torch.float32).mean(dim=(1, 2))
    return (coverage / reference).clamp(0, 1)


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

    logger.info("Loaded frozen teacher: %s (input channels = %d)", checkpoint_path, ch)
    return teacher


def head_outputs(preds) -> dict:
    """The head's prediction dict, whether the model ran in train or eval mode."""
    return preds[1] if isinstance(preds, (list, tuple)) else preds


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
