import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.utils.ops import non_max_suppression

from dsec_det.label import CLASSES

import log
from train_rgb_student import (
    create_rgb_test_loader,
    extract_trainable_model,
    select_device,
)

# OpenCV draws in BGR.
PREDICTION_COLOUR = (0, 0, 255)
GROUND_TRUTH_COLOUR = (0, 255, 0)


def pad_to_stride(imgs, stride=32):
    """Pad the bottom/right so height and width divide by 32 (1080 does not)."""
    height, width = imgs.shape[-2:]
    pad_height = (-height) % stride
    pad_width = (-width) % stride
    return torch.nn.functional.pad(imgs, (0, pad_width, 0, pad_height))


def load_student_model(checkpoint_path, weights_path, device):
    """Rebuild the student and restore the trained weights in eval mode."""
    model = extract_trainable_model(weights_path, device=str(device))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handles both the {'model_state_dict': ...} wrapper and a bare state_dict.
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model, imgs, device, conf=0.25, iou=0.45):
    """Forward pass plus NMS. One [N, 6] tensor (xyxy, conf, cls) per image."""
    raw = model(pad_to_stride(imgs.to(device)))
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    return non_max_suppression(raw, conf_thres=conf, iou_thres=iou)


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


def visualise_batches(test_loader, model, device, output_directory, num_batches=1):
    """Annotate the first few test batches and save each image."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = []
    for batch_index, (imgs, targets) in enumerate(test_loader):
        if batch_index >= num_batches:
            break
        detections = predict_batch(model, imgs, device)
        height, width = imgs.shape[-2:]
        for i in range(len(imgs)):
            truth_boxes = ground_truth_boxes(targets, i, width, height)
            image = annotate_sample(imgs[i], detections[i], truth_boxes)
            path = output_directory / f"batch{batch_index}_image{i}.png"
            cv2.imwrite(str(path), image)
            paths.append(path)
            print(f"{path.name}: {len(detections[i])} predicted, "
                  f"{len(truth_boxes)} ground truth")

    print(f"Saved {len(paths)} images to {output_directory}")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Draw predicted and ground-truth boxes on test images."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("best_student_yolo.pt"))
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=1)
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device}")
    model = load_student_model(args.checkpoint, args.weights, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    test_loader = create_rgb_test_loader(args.dataset_path, args.batch_size)
    output_directory = log.create_log_directory() / "predictions"
    visualise_batches(test_loader, model, device, output_directory, args.num_batches)


if __name__ == "__main__":
    main()