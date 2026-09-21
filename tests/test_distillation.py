from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from distillation import DISTILL_LEVEL, feature_distillation_loss, head_outputs, load_teacher


def predictions(values, key="one2one"):
    return {key: {"feats": [torch.full((1, 2, 4, 4), value) for value in values]}}


class FeatureDistillationLossTests(unittest.TestCase):
    def test_identical_features_give_zero_loss(self):
        loss = feature_distillation_loss(predictions([1.0, 2.0]), predictions([1.0, 2.0]))
        self.assertEqual(loss.item(), 0.0)

    def test_different_features_give_positive_loss(self):
        loss = feature_distillation_loss(predictions([0.0, 0.0]), predictions([1.0, 1.0]))
        self.assertGreater(loss.item(), 0.0)

    def test_only_the_selected_level_is_compared(self):
        # P3 matches, the coarser levels do not: the loss must ignore them.
        student = predictions([1.0, 5.0, 9.0])
        teacher = predictions([1.0, 0.0, 0.0])
        self.assertEqual(DISTILL_LEVEL, 0)
        self.assertEqual(feature_distillation_loss(student, teacher).item(), 0.0)
        self.assertGreater(feature_distillation_loss(student, teacher, level=1).item(), 0.0)

    def test_eval_mode_teacher_output_is_unwrapped(self):
        eval_output = (torch.zeros(1, 300, 6), predictions([1.0, 2.0]))
        self.assertIs(head_outputs(eval_output), eval_output[1])
        loss = feature_distillation_loss(predictions([1.0, 2.0]), eval_output)
        self.assertEqual(loss.item(), 0.0)

    def test_no_gradient_flows_into_the_teacher(self):
        student = predictions([0.0])
        student["one2one"]["feats"][0].requires_grad_(True)
        teacher_feature = torch.full((1, 2, 4, 4), 1.0, requires_grad=True)
        teacher = {"one2one": {"feats": [teacher_feature]}}

        feature_distillation_loss(student, teacher).backward()

        self.assertIsNotNone(student["one2one"]["feats"][0].grad)
        self.assertIsNone(teacher_feature.grad)


class LoadTeacherTests(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.checkpoint = Path(self.temp_dir.name) / "best_teacher_yolo.pt"

    def test_missing_checkpoint_is_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "Teacher checkpoint not found"):
            load_teacher(self.checkpoint, torch.device("cpu"), self.logger)
        self.logger.error.assert_called_once()

    @patch("distillation.build_model")
    def test_teacher_is_frozen_and_in_eval_mode(self, build_model):
        model = torch.nn.Linear(2, 2)
        model.train()
        build_model.return_value = model
        torch.save({"model_state_dict": model.state_dict()}, self.checkpoint)

        teacher = load_teacher(self.checkpoint, torch.device("cpu"), self.logger)

        build_model.assert_called_once_with(5)
        self.assertFalse(teacher.training)
        self.assertFalse(any(parameter.requires_grad for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
