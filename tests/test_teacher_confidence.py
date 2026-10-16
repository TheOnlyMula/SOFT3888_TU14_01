"""Focused acceptance tests for the teacher-confidence score in Issue #57."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import yaml

from distillation import TEACHER_CONFIDENCE_THRESHOLD, teacher_confidence


def detections_with_scores(*score_rows):
    """Build made-up YOLO end-to-end detections with shape [batch, boxes, 6]."""
    predictions = torch.zeros(len(score_rows), max(map(len, score_rows)), 6)
    for image_index, scores in enumerate(score_rows):
        for box_index, score in enumerate(scores):
            left = box_index * 20
            predictions[image_index, box_index] = torch.tensor(
                [left, 0, left + 10, 10, score, 0]
            )
    return predictions


class TeacherConfidenceTests(unittest.TestCase):
    def test_none_one_and_several_retained_detections_per_frame(self):
        predictions = detections_with_scores(
            [0.10, 0.24],
            [0.80, 0.10],
            [0.30, 0.60, 0.90],
        )

        scores = teacher_confidence(predictions)

        self.assertEqual(scores.shape, (3,))
        torch.testing.assert_close(
            scores, torch.tensor([0.0, 0.8, 0.6]), rtol=0, atol=1e-6
        )

    def test_eval_output_uses_detection_tensor(self):
        predictions = detections_with_scores([0.70, 0.10])
        eval_output = (predictions, {"one2many": {"feats": []}})

        scores = teacher_confidence(eval_output)

        torch.testing.assert_close(scores, torch.tensor([0.7]), rtol=0, atol=1e-6)

    def test_detection_at_threshold_is_retained(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                predictions = detections_with_scores([0.24, 0.25, 0.26]).to(dtype)

                scores = teacher_confidence(predictions, threshold=0.25)

                torch.testing.assert_close(
                    scores, torch.tensor([0.255], dtype=dtype), rtol=0,
                    atol=1e-3 if dtype == torch.float16 else 1e-6,
                )


class TeacherConfidenceTrainingTests(unittest.TestCase):
    def test_training_uses_one_teacher_forward_without_gradients(self):
        from torch.utils.data import DataLoader

        import training
        from dataset_loader import yolo_collate_fn
        from utils import build_model, extract_trainable_model

        sample = (
            torch.rand(5, 64, 64),
            torch.tensor([[2.0, 0.4, 0.5, 0.3, 0.4]]),
        )
        loader = DataLoader([sample], batch_size=1, collate_fn=yolo_collate_fn)
        student = extract_trainable_model(ch=3, weights_path=None, device="cpu")
        teacher = build_model(5).eval().requires_grad_(False)
        original_forward = teacher.forward
        original_confidence = training.losses.teacher_confidence
        forward_results = []
        forward_grad_modes = []
        confidence_calls = []

        def record_forward(images):
            forward_grad_modes.append(torch.is_grad_enabled())
            result = original_forward(images)
            forward_results.append(result)
            return result

        def record_confidence(predictions, threshold):
            confidence_calls.append((predictions, torch.is_grad_enabled(), threshold))
            return original_confidence(predictions, threshold)

        validation_results = {
            "loss": 1.0, "box_loss": 0.0, "cls_loss": 0.0, "l1_loss": 0.0,
            "precision": 0.0, "recall": 0.0, "map50": 0.0,
            "map50_95": 0.1, "agnostic_map50": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(teacher, "forward", side_effect=record_forward) as forward, \
                patch.object(training.losses, "teacher_confidence", side_effect=record_confidence) as confidence, \
                patch.object(training.losses, "feature_distillation_loss", wraps=training.losses.feature_distillation_loss) as distillation_loss, \
                patch.object(training.loop, "validate_model", return_value=validation_results):
            training.train_model(
                loader, loader, student,
                torch.optim.SGD(student.parameters(), lr=0.01),
                torch.device("cpu"), MagicMock(),
                total_epochs=1,
                checkpoint_path=Path(directory) / "best.pt",
                latest_path=Path(directory) / "latest.pt",
                teacher=teacher,
                lambda_distill=1.0,
                nominal_batch_size=1,
                warmup_epochs=0,
            )

        self.assertEqual(forward.call_count, 1)
        self.assertEqual(forward_grad_modes, [False])
        self.assertEqual(confidence.call_count, 1)
        self.assertEqual(distillation_loss.call_count, 1)
        self.assertIs(confidence_calls[0][0], forward_results[0])
        self.assertIs(distillation_loss.call_args.args[1], forward_results[0])
        self.assertFalse(confidence_calls[0][1])
        self.assertEqual(confidence_calls[0][2], TEACHER_CONFIDENCE_THRESHOLD)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_distillation_run_records_threshold_in_config_yaml(self):
        import training

        loader = SimpleNamespace(
            dataset=[0], batch_size=1, sampler=[], num_workers=0,
            pin_memory=False, persistent_workers=False,
        )
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            training.run,
            select_device=MagicMock(return_value=torch.device("cpu")),
            init_seeds=MagicMock(),
            create_dataloaders=MagicMock(return_value=(loader, loader)),
            extract_trainable_model=MagicMock(),
            load_teacher=MagicMock(),
            create_optimiser=MagicMock(return_value=object()),
            train_model=MagicMock(return_value=Path(directory) / "best.pt"),
        ), patch.object(training.run.log, "create_log_directory", return_value=Path(directory)), \
                patch.object(training.run.log, "create_logger", return_value=MagicMock()), \
                patch.object(training.run.log, "log_command"), \
                patch("sys.argv", ["train", "--dataset-path", "fake", "--skip-test",
                                   "--teacher-checkpoint", "teacher.pt"]):
            training.main(role="distill", rgb_only=True)
            config = yaml.safe_load((Path(directory) / "config.yaml").read_text())

        self.assertEqual(config["Teacher Confidence Threshold"], TEACHER_CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
