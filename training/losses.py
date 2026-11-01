"""The losses of one training batch: detection, plus distillation when there is a teacher."""
import torch

from .distillation import (
    teacher_confidence,
    distilled_features,
    event_density,
    feature_distillation_loss,
    object_distillation_loss,
    object_region_mask,
    quality_weight,
)


def batch_losses(
    model: torch.nn.Module,
    batch_data: dict,
    teacher: torch.nn.Module | None = None,
    distill_mode: str = "unweighted",
    event_coverage: torch.Tensor | None = None,
):
    """Run the student, and the teacher if there is one, on a batch and return its losses.

    `event_coverage` is each image's event-window coverage, needed in "object-quality" mode.
    Returns (loss, loss_items, distill_loss, quality): the detection loss and its parts, the batch
    distillation loss (None without a teacher), and in "object-quality" mode each image's s_e,
    c_t and w (None otherwise).
    """
    if teacher is None:
        loss, loss_items = model(batch_data)
        return loss, loss_items, None, None

    # The student sees RGB only; the teacher sees the same frames plus events.
    student_batch = {**batch_data, "img": batch_data["img"][:, :3]}
    student_preds = model(student_batch["img"])
    loss, loss_items = model.loss(student_batch, student_preds)
    with torch.no_grad():
        teacher_preds = teacher(batch_data["img"])

    if distill_mode == "unweighted":
        return loss, loss_items, feature_distillation_loss(student_preds, teacher_preds), None

    mask = object_region_mask(
        batch_data["batch_idx"],
        batch_data["bboxes"],
        batch_data["img"].shape[0],
        distilled_features(student_preds, teacher_preds)[0].shape[-2:],
    )
    per_image_loss = object_distillation_loss(student_preds, teacher_preds, mask)
    quality = None
    if distill_mode == "object-quality":
        # The only difference from "object", which is this mode with every w = 1. The mean below
        # still runs over every image, so a down-weighted image does not pass its share to the others.
        s_e = event_density(event_coverage.to(per_image_loss.device))
        c_t = teacher_confidence(teacher_preds)
        quality = {"s_e": s_e, "c_t": c_t, "w": quality_weight(s_e, c_t)}
        per_image_loss = quality["w"] * per_image_loss
    return loss, loss_items, per_image_loss.mean(), quality
