import unittest

import numpy as np
import torch

from predict_visualise import (
    GROUND_TRUTH_COLOUR,
    PREDICTION_COLOUR,
    annotate_sample,
    class_name,
    ground_truth_boxes,
    pad_to_stride,
)


class DrawingTests(unittest.TestCase):
    def test_dsec_frame_is_padded_to_a_multiple_of_32(self):
        padded = pad_to_stride(torch.ones(2, 3, 1080, 1440))
        self.assertEqual(tuple(padded.shape), (2, 3, 1088, 1440))

    def test_padding_leaves_the_original_pixels_in_place(self):
        imgs = torch.ones(1, 3, 30, 30)
        padded = pad_to_stride(imgs)
        self.assertTrue(torch.equal(padded[:, :, :30, :30], imgs))

    def test_class_id_outside_the_dsec_names_does_not_crash(self):
        self.assertEqual(class_name(79), "79")

    def test_targets_become_pixel_corners_for_the_right_image(self):
        targets = torch.tensor([
            [0.0, 2.0, 0.25, 0.5, 0.25, 0.5],
            [1.0, 3.0, 0.5, 0.5, 1.0, 1.0],
        ])
        self.assertEqual(ground_truth_boxes(targets, 0, 16, 16),
                         [(2, 2.0, 4.0, 6.0, 12.0)])
        self.assertEqual(ground_truth_boxes(torch.empty((0, 6)), 0, 16, 16), [])

    def test_annotate_draws_both_colours(self):
        detections = torch.tensor([[5.0, 5.0, 25.0, 25.0, 0.9, 0.0]])
        image = annotate_sample(torch.zeros(3, 64, 64), detections,
                                [(1, 35.0, 35.0, 55.0, 55.0)])
        self.assertTrue((image == np.array(PREDICTION_COLOUR)).all(axis=2).any())
        self.assertTrue((image == np.array(GROUND_TRUTH_COLOUR)).all(axis=2).any())

    def test_annotate_handles_no_detections(self):
        image = annotate_sample(torch.zeros(3, 32, 32), torch.empty((0, 6)), [])
        self.assertEqual(image.shape, (32, 32, 3))


if __name__ == "__main__":
    unittest.main()