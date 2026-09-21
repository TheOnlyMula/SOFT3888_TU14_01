"""Unweighted feature distillation from a frozen privileged teacher to the RGB student.

The teacher sees RGB + event channels; the student only ever sees the RGB channels of the
same frame. This is the basic (unweighted) baseline required before the object-aware and
quality-weighted variants: every spatial location counts equally, with no object mask and
no event-quality weight.
"""
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from utils import build_model

# Neck features fed to the detection head. The student and teacher share this architecture,
# so the feature maps already match and need no projection layer.
DISTILL_TARGET = "one2one"

#Level 0  is P3 (Stride 8). Levels 1 and 2 are P4 (stride 16) and P5 (stride 32).
DISTILL_LEVEL = 0


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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    teacher = build_model(ch)
    teacher.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    teacher = teacher.to(device).eval().requires_grad_(False)

    logger.info("Loaded frozen teacher: %s (input channels = %d)", checkpoint_path, ch)
    return teacher


def head_outputs(preds) -> dict:
    """The head's prediction dict, whether the model ran in train or eval mode."""
    return preds[1] if isinstance(preds, (list, tuple)) else preds


def feature_distillation_loss(student_preds, teacher_preds, level: int = DISTILL_LEVEL) -> torch.Tensor:
    """Smooth L1 distance between the student's and teacher's features at one neck level.

    The teacher side is detached, so no gradient reaches the frozen teacher. Every spatial
    location counts equally as the object mask and event-quality weight come later.
    """
    student_feature = head_outputs(student_preds)[DISTILL_TARGET]["feats"][level]
    teacher_feature = head_outputs(teacher_preds)[DISTILL_TARGET]["feats"][level]

    return F.smooth_l1_loss(student_feature, teacher_feature.detach())
