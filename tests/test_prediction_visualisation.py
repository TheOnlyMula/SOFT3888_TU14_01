import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from predict_visualise import (
    GROUND_TRUTH_COLOUR,
    PREDICTION_COLOUR,
    annotate_sample,
    class_name,
    ground_truth_boxes,
    load_student_model,
    pad_to_stride,
    predict_batch,
)


class FakeModel(torch.nn.Module):
    """Records eval/grad state and returns a fixed raw prediction."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.observed = []

    def forward(self, imgs):
        self.observed.append((self.training, torch.is_grad_enabled(), tuple(imgs.shape)))
        # (batch, 4 + classes, anchors) with one confident anchor.
        prediction = torch.zeros(len(imgs), 5, 2)
        prediction[:, :4, 0] = torch.tensor([10.0, 10.0, 8.0, 8.0])
        prediction[:, 4, 0] = 0.9
        return prediction


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


class InferenceTests(unittest.TestCase):
    def test_forward_pass_is_padded_and_runs_without_gradients(self):
        model = FakeModel()
        model.eval()
        detections = predict_batch(model, torch.zeros(2, 3, 30, 30), torch.device("cpu"))
        training_mode, grad_enabled, shape = model.observed[0]
        self.assertFalse(training_mode)
        self.assertFalse(grad_enabled)
        self.assertEqual(shape, (2, 3, 32, 32))
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].shape[1], 6)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.reference = FakeModel()
        with torch.no_grad():
            self.reference.weight.fill_(7.0)

    def _load(self, path):
        fresh = FakeModel()
        fresh.train()
        with patch("predict_visualise.extract_trainable_model", return_value=fresh):
            return load_student_model(path, "dummy.pt", torch.device("cpu"))

    def test_training_checkpoint_is_restored_in_eval_mode(self):
        path = self.directory / "best_student_yolo.pt"
        torch.save({"epoch": 2, "model_state_dict": self.reference.state_dict(),
                    "optimiser_state_dict": {}, "val_loss": 0.5}, path)
        model = self._load(path)
        self.assertEqual(model.weight.item(), 7.0)
        self.assertFalse(model.training)

    def test_bare_state_dict_is_also_accepted(self):
        path = self.directory / "bare.pt"
        torch.save(self.reference.state_dict(), path)
        self.assertEqual(self._load(path).weight.item(), 7.0)


if __name__ == "__main__":
    unittest.main()