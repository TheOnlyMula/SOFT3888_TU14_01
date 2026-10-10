from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

import training
from distillation import (
    DISTILL_LEVEL,
    DISTILL_TARGET,
    feature_distillation_loss,
    head_outputs,
    load_teacher,
)
from utils import build_model


def predictions(values, key=DISTILL_TARGET):
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
        student[DISTILL_TARGET]["feats"][0].requires_grad_(True)
        teacher_feature = torch.full((1, 2, 4, 4), 1.0, requires_grad=True)
        teacher = {DISTILL_TARGET: {"feats": [teacher_feature]}}

        feature_distillation_loss(student, teacher).backward()

        self.assertIsNotNone(student[DISTILL_TARGET]["feats"][0].grad)
        self.assertIsNone(teacher_feature.grad)

    def test_distillation_gradient_reaches_the_real_student(self):
        # The YOLO26 head detaches the one2one features in training mode, so a hand-built
        # prediction dict cannot show whether the loss actually trains the student.
        student = build_model(3).train()
        teacher = build_model(5).eval().requires_grad_(False)
        images = torch.rand(1, 5, 64, 64)

        student_preds = student(images[:, :3])
        with torch.no_grad():
            teacher_preds = teacher(images)
        loss = feature_distillation_loss(student_preds, teacher_preds)

        self.assertTrue(loss.requires_grad)
        loss.backward()
        stem_grad = student.model[0].conv.weight.grad
        self.assertIsNotNone(stem_grad)
        self.assertGreater(stem_grad.abs().sum().item(), 0.0)


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


class DistillationSeedTests(unittest.TestCase):
    def student_init_draw(self, extra_args, role):
        """Run main with real seeding and return the random draw the student is built from."""
        draws = []

        def build_student(**kwargs):
            draws.append(torch.rand(8))
            return MagicMock()

        def build_teacher(*args, **kwargs):
            torch.randn(10_000)  
            return MagicMock()

        loader = MagicMock()
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            training,
            select_device=MagicMock(return_value=torch.device("cpu")),
            create_dataloaders=MagicMock(return_value=(loader, loader)),
            extract_trainable_model=MagicMock(side_effect=build_student),
            load_teacher=MagicMock(side_effect=build_teacher),
            create_optimiser=MagicMock(),
            train_model=MagicMock(return_value=Path(directory) / "best.pt"),
        ), patch.object(training.log, "create_log_directory", return_value=Path(directory)), \
                patch.object(training.log, "create_logger", return_value=MagicMock()), \
                patch.object(training.log, "log_command"), \
                patch.object(training.log, "log_model_config"), \
                patch("sys.argv", ["train", "--dataset-path", "fake", "--skip-test", *extra_args]):
            training.main(role=role, rgb_only=True)
        return draws[0]

    def test_distilled_student_starts_from_the_same_weights_as_the_rgb_baseline(self):
        baseline = self.student_init_draw([], role="student")
        distilled = self.student_init_draw(["--teacher-checkpoint", "teacher.pt"], role="distill")
        self.assertTrue(torch.equal(baseline, distilled))


if __name__ == "__main__":
    unittest.main()
