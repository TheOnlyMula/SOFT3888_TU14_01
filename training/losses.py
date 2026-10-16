"""The losses of one training batch: detection, plus distillation when there is a teacher."""
import torch

from distillation import (
    TEACHER_CONFIDENCE_THRESHOLD,
    teacher_confidence,
    distilled_features,
    event_density,
    feature_distillation_loss,
    object_distillation_loss,
    object_region_mask,
    quality_weight,
    quality_weighted_loss,
)


def batch_losses(
    model: torch.nn.Module,
    batch_data: dict,
    teacher: torch.nn.Module | None = None,
    distill_mode: str = "unweighted",
):
    """Run the student, and the teacher if there is one, on a batch and return its losses.

    Returns (loss, loss_items, distill_loss, weight): the detection loss and its parts, the batch
    distillation loss (None without a teacher), and each image's quality weight w (None outside
    "object-quality" mode).
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
        teacher_confidences = teacher_confidence(
            teacher_preds,
            threshold=TEACHER_CONFIDENCE_THRESHOLD,
        )

    weight = None
    if distill_mode in ("object", "object-quality"):
        mask = object_region_mask(
            batch_data["batch_idx"],
            batch_data["bboxes"],
            batch_data["img"].shape[0],
            distilled_features(student_preds, teacher_preds)[0].shape[-2:],
        )
        per_image_loss = object_distillation_loss(student_preds, teacher_preds, mask)
        if distill_mode == "object-quality":
            # The only difference from "object": each image's loss is scaled by its w
            weight = quality_weight(event_density(batch_data["img"]), teacher_confidences)
            distill_loss = quality_weighted_loss(per_image_loss, weight)
        else:
            distill_loss = per_image_loss.mean()
    else:
        distill_loss = feature_distillation_loss(student_preds, teacher_preds)
    return loss, loss_items, distill_loss, weight
