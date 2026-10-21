from pathlib import Path
import random
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from loaders import augment
from loaders.augment import EVENT_PAD, RGB_PAD, augment_sample
from loaders.dataset_loader import DSECFusionDataset

HEIGHT, WIDTH = 64, 80
BOX = np.array([[20.0, 16.0, 44.0, 40.0]])


def marked_sample():
    """A white RGB square and an event square in the same place, with a box around them."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    events = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)
    x1, y1, x2, y2 = BOX[0].astype(int)
    image[y1:y2, x1:x2] = 255
    events[y1:y2, x1:x2] = 1.0
    return image, events, BOX.copy(), np.array([2])


def fixed_draws(flip, scale, shift_x, shift_y):
    """Patch augment's random draws: no hue shift, saturation gain 1, then scale and shifts."""
    return (
        patch("loaders.augment.random.random", return_value=0.0 if flip else 1.0),
        patch("loaders.augment.random.uniform", side_effect=[0.0, 1.0, scale, shift_x, shift_y]),
    )


def hue_of(rgb_pixel):
    pixel = np.array([[rgb_pixel]], dtype=np.uint8)
    return int(cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0, 0])


class AugmentSampleTests(unittest.TestCase):
    def test_rgb_events_and_boxes_stay_aligned(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                random.seed(seed)
                image, events, boxes, classes = augment_sample(*marked_sample())

                rgb_mask = image[..., 0] > 127
                event_mask = events[..., 0] > 0.5
                # uint8 and float interpolation can round a half-covered edge pixel differently
                self.assertLess(np.mean(rgb_mask != event_mask), 0.01)
                self.assertEqual(len(boxes), 1)
                self.assertEqual(classes.tolist(), [2])

                ys, xs = np.nonzero(rgb_mask)
                marker = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
                np.testing.assert_allclose(boxes[0], marker, atol=1.5)

    def test_flip_mirrors_boxes(self):
        flip, uniform = fixed_draws(flip=True, scale=1.0, shift_x=0.0, shift_y=0.0)
        with flip, uniform:
            _, events, boxes, _ = augment_sample(*marked_sample())
        np.testing.assert_allclose(boxes[0], [WIDTH - 44.0, 16.0, WIDTH - 20.0, 40.0])
        self.assertEqual(events[20, WIDTH - 30, 0], 1.0)
        self.assertEqual(events[20, 30, 0], 0.0)

    def test_uncovered_pixels_are_grey_rgb_and_empty_events(self):
        # Zooming out to 75% leaves a border with no source pixels
        flip, uniform = fixed_draws(flip=False, scale=0.75, shift_x=0.0, shift_y=0.0)
        with flip, uniform:
            image, events, _, _ = augment_sample(
                np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8),
                np.ones((HEIGHT, WIDTH, 2), dtype=np.float32),
                BOX.copy(),
                np.array([2]),
            )
        self.assertEqual(image[0, 0].tolist(), [RGB_PAD] * 3)
        self.assertEqual(events[0, 0].tolist(), [EVENT_PAD] * 2)
        self.assertEqual(events[HEIGHT // 2, WIDTH // 2].tolist(), [1.0, 1.0])

    def test_boxes_shifted_out_of_the_image_are_dropped_with_their_class(self):
        flip, uniform = fixed_draws(flip=False, scale=1.0, shift_x=-augment.TRANSLATE, shift_y=0.0)
        boxes = np.array([[0.0, 10.0, 6.0, 30.0], [40.0, 10.0, 70.0, 30.0]])
        with flip, uniform:
            _, _, kept, classes = augment_sample(
                np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), None, boxes, np.array([1, 5])
            )
        self.assertEqual(classes.tolist(), [5])
        np.testing.assert_allclose(kept, [[32.0, 10.0, 62.0, 30.0]])

    def test_boxes_cut_into_thin_slivers_are_dropped(self):
        # The right box is clipped to 3 x 360 px (ratio 120) by the image edge. Its visible area
        # is still 37%, so only the aspect-ratio check removes it.
        height, width = 400, 640
        boxes = np.array([[100.0, 20.0, 200.0, 380.0], [637.0, 20.0, 645.0, 380.0]])
        flip, uniform = fixed_draws(flip=False, scale=1.0, shift_x=0.0, shift_y=0.0)
        with flip, uniform:
            _, _, kept, classes = augment_sample(
                np.zeros((height, width, 3), dtype=np.uint8), None, boxes, np.array([2, 4])
            )
        self.assertEqual(classes.tolist(), [2])
        np.testing.assert_allclose(kept, boxes[:1])

    def test_hue_shifts_every_colour_by_the_same_amount(self):
        # Under the old multiplicative formula red (hue 0) never moved while blue did
        red, blue = (255, 0, 0), (0, 0, 255)
        image = np.array([[red, blue]], dtype=np.uint8)
        with patch("loaders.augment.random.uniform", side_effect=[augment.HUE_GAIN, 1.0]):
            shifted = augment.jitter_hue_saturation(image)
        red_shift = hue_of(shifted[0, 0]) - hue_of(red)
        blue_shift = hue_of(shifted[0, 1]) - hue_of(blue)
        self.assertGreaterEqual(red_shift, 1)
        self.assertLessEqual(abs(red_shift - blue_shift), 1)

    def test_colour_jitter_keeps_brightness(self):
        image = np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        random.seed(0)
        jittered = augment.jitter_hue_saturation(image)
        # HSV value is the max channel; colour conversion rounding moves it by at most a few levels
        self.assertLessEqual(np.abs(jittered.max(-1).astype(int) - image.max(-1)).max(), 3)


class DatasetAugmentationTests(unittest.TestCase):
    @patch("loaders.dataset_loader.load_causal_events")
    @patch("loaders.dataset_loader.DSECDet")
    def test_rgb_student_and_teacher_see_the_same_augmented_rgb(self, dataset_class, events):
        base = dataset_class.return_value
        base.get_image.return_value = np.random.default_rng(1).integers(
            0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8
        )
        base.get_tracks.return_value = [{"class_id": 2, "x": 20.0, "y": 16.0, "w": 24.0, "h": 24.0}]
        events.return_value = {"x": np.array([30]), "y": np.array([20]), "p": np.array([1])}

        student = DSECFusionDataset(Path("fake"), rgb_only=True, augment=True)
        teacher = DSECFusionDataset(Path("fake"), rgb_only=False, augment=True)
        for seed in range(5):
            random.seed(seed)
            student_image, student_targets = student[0]
            random.seed(seed)
            teacher_image, teacher_targets = teacher[0]
            self.assertTrue(torch.equal(student_image, teacher_image[:3]))
            self.assertTrue(torch.equal(student_targets, teacher_targets))

    @patch("loaders.dataset_loader.DSECDet")
    def test_augmentation_is_off_by_default(self, dataset_class):
        base = dataset_class.return_value
        image = np.random.default_rng(2).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        base.get_image.return_value = image
        base.get_tracks.return_value = []
        first, _ = DSECFusionDataset(Path("fake"), rgb_only=True)[0]
        second, _ = DSECFusionDataset(Path("fake"), rgb_only=True)[0]
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
