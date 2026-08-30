import cv2
import numpy as np
import torch

from dsec_det.label import CLASSES

# OpenCV draws in BGR.
PREDICTION_COLOUR = (0, 0, 255)
GROUND_TRUTH_COLOUR = (0, 255, 0)


def pad_to_stride(imgs, stride=32):
    """Pad the bottom/right so height and width divide by 32 (1080 does not)."""
    height, width = imgs.shape[-2:]
    pad_height = (-height) % stride
    pad_width = (-width) % stride
    return torch.nn.functional.pad(imgs, (0, pad_width, 0, pad_height))


def class_name(class_id):
    """The head is still 80-class, so ids can fall outside the DSEC names."""
    return CLASSES[class_id] if 0 <= class_id < len(CLASSES) else str(class_id)


def ground_truth_boxes(targets, image_index, width, height):
    """Turn collated [batch_idx, cls, xc, yc, w, h] rows into pixel corners."""
    boxes = []
    rows = targets[targets[:, 0] == image_index]
    for _, class_id, x_centre, y_centre, norm_w, norm_h in rows.tolist():
        box_w = norm_w * width
        box_h = norm_h * height
        x1 = x_centre * width - box_w / 2
        y1 = y_centre * height - box_h / 2
        boxes.append((int(class_id), x1, y1, x1 + box_w, y1 + box_h))
    return boxes


def draw_box(image, box, label, colour):
    x1, y1, x2, y2 = [int(value) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(image, label, (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)


def annotate_sample(img_tensor, detections, truth_boxes):
    """Return the image with ground truth (green) and predictions (red) on it."""
    image = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = np.ascontiguousarray(image)

    for class_id, x1, y1, x2, y2 in truth_boxes:
        draw_box(image, (x1, y1, x2, y2), class_name(class_id), GROUND_TRUTH_COLOUR)

    for x1, y1, x2, y2, confidence, class_id in detections.tolist():
        label = f"{class_name(int(class_id))} {confidence:.2f}"
        draw_box(image, (x1, y1, x2, y2), label, PREDICTION_COLOUR)

    return image