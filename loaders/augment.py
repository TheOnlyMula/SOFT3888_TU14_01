"""Training-time augmentation shared by every model.

Geometric changes (flip, scale, shift) use one random draw for the RGB pixels, the event pixels
and the boxes, so the teacher's events stay aligned with the RGB the student sees. Colour changes
touch RGB only and leave brightness alone: brightness shifts belong to the degradation-augmentation
baseline, not to every student.

Random draws come from Python's `random` module, which PyTorch seeds per DataLoader worker. Every
sample makes the same draws with or without events, so the RGB student and the distilled student
see identical augmented RGB for the same seed.
"""
import random

import cv2
import numpy as np

FLIP_PROB = 0.5
SCALE = 0.5  # zoom drawn from [1 - SCALE, 1 + SCALE], as Ultralytics' default
TRANSLATE = 0.1  # shift drawn from +/- this fraction of the width and height
HUE_GAIN = 0.015  # Ultralytics hsv_h: hue shifts by up to this fraction of the colour wheel
SATURATION_GAIN = 0.7  # Ultralytics hsv_s

# Uncovered RGB pixels get Ultralytics' grey. Uncovered event pixels stay 0: in a histogram 0
# means "no events", and any other value would invent events along the border.
RGB_PAD = 114
EVENT_PAD = 0.0

# A box survives the warp only if both sides stay above MIN_BOX_SIDE pixels, at least
# MIN_VISIBLE_AREA of its warped area is still inside the image, and it is not a sliver more
# than MAX_ASPECT_RATIO times longer than it is wide (Ultralytics' box_candidates).
MIN_BOX_SIDE = 2.0
MIN_VISIBLE_AREA = 0.1
MAX_ASPECT_RATIO = 100.0


def jitter_hue_saturation(image: np.ndarray) -> np.ndarray:
    """Randomly shift hue and scale saturation of an RGB uint8 image, brightness is unchanged.
    """
    hue_shift = random.uniform(-HUE_GAIN, HUE_GAIN) * 180
    saturation_gain = random.uniform(1 - SATURATION_GAIN, 1 + SATURATION_GAIN)

    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    levels = np.arange(256, dtype=np.float32)
    hue = cv2.LUT(hue, ((levels + hue_shift) % 180).astype(np.uint8))
    saturation = cv2.LUT(saturation, np.clip(levels * saturation_gain, 0, 255).astype(np.uint8))
    return cv2.cvtColor(cv2.merge((hue, saturation, value)), cv2.COLOR_HSV2RGB)


def augment_sample(
    image: np.ndarray,
    events: np.ndarray | None,
    boxes: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Augment one training sample.

    Args:
        image: H x W x 3 uint8 RGB image.
        events: H x W x C float32 event histogram, or None for RGB-only samples.
        boxes: N x 4 boxes as pixel x1, y1, x2, y2.
        classes: N class ids.

    Returns:
        The augmented image, events, boxes and classes. Boxes that leave the image are dropped
        together with their classes.
    """
    height, width = image.shape[:2]
    boxes = boxes.astype(np.float32).reshape(-1, 4)

    image = jitter_hue_saturation(image)

    if random.random() < FLIP_PROB:
        image = image[:, ::-1]
        if events is not None:
            events = events[:, ::-1]
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

    scale = random.uniform(1 - SCALE, 1 + SCALE)
    offset_x = (1 - scale) * width / 2 + random.uniform(-TRANSLATE, TRANSLATE) * width
    offset_y = (1 - scale) * height / 2 + random.uniform(-TRANSLATE, TRANSLATE) * height
    matrix = np.array([[scale, 0, offset_x], [0, scale, offset_y]], dtype=np.float32)

    image = cv2.warpAffine(
        np.ascontiguousarray(image), matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderValue=(RGB_PAD, RGB_PAD, RGB_PAD),
    )
    if events is not None:
        channels = events.shape[2]
        events = cv2.warpAffine(
            np.ascontiguousarray(events), matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderValue=EVENT_PAD,
        ).reshape(height, width, channels) 

    warped = boxes * scale + np.array([offset_x, offset_y, offset_x, offset_y], dtype=np.float32)
    clipped = warped.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, height)

    clipped_w = clipped[:, 2] - clipped[:, 0]
    clipped_h = clipped[:, 3] - clipped[:, 1]
    warped_area = (warped[:, 2] - warped[:, 0]) * (warped[:, 3] - warped[:, 1])
    aspect_ratio = np.maximum(clipped_w / (clipped_h + 1e-16), clipped_h / (clipped_w + 1e-16))
    keep = (
        (clipped_w > MIN_BOX_SIDE)
        & (clipped_h > MIN_BOX_SIDE)
        & (clipped_w * clipped_h > MIN_VISIBLE_AREA * warped_area)
        & (aspect_ratio < MAX_ASPECT_RATIO)
    )
    return image, events, clipped[keep], np.asarray(classes).reshape(-1)[keep]
