from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import torch
from torch.utils.data import DataLoader, SequentialSampler

from dataset_loader import yolo_collate_fn
from dataset_loader import create_rgb_test_loader
from evaluate import evaluate_test_model, format_results_table
from training import train_model
import training


class BatchLossModel(torch.nn.Module):
    """Small real module that records eval and gradient state."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.bn = torch.nn.BatchNorm2d(3)
        self.dropout = torch.nn.Dropout()
        self.observed = []

    def forward(self, x):
        if isinstance(x, dict):  # training path: model(batch) returns the loss
            return self.loss(x)
        self.dropout(self.bn(x))
        return x

    def loss(self, batch, preds=None):
        if preds is None:
            self.dropout(self.bn(batch["img"]))
        self.observed.append(
            (
                self.training,
                self.bn.training,
                self.dropout.training,
                torch.is_grad_enabled(),
                batch["cls"].shape,
                batch["bboxes"].shape,
                batch["batch_idx"].shape,
            )
        )
        count = batch["img"].size(0)
        return torch.stack((self.weight * count, self.weight * count / 2)), {
            "box_loss": torch.tensor(0.3),
            "cls_loss": torch.tensor(0.2),
            "l1_loss": torch.tensor(0.1),
        }


def val_results(loss):
    return {
        "loss": loss,
        "box_loss": 0.0,
        "cls_loss": 0.0,
        "l1_loss": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
        "agnostic_map50": 0.0,
    }


def make_samples(count=5, empty_targets=False):
    return [
        (
            torch.ones(3, 8, 8) * index,
            torch.empty((0, 5))
            if empty_targets
            else torch.tensor([[1.0, 0.5, 0.5, 0.2, 0.2]]),
        )
        for index in range(count)
    ]


def make_loader(count=5, empty_targets=False):
    return DataLoader(
        make_samples(count, empty_targets),
        batch_size=4,
        collate_fn=yolo_collate_fn,
        shuffle=False,
    )


class PostTrainingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.checkpoint = self.directory / "best_student_yolo.pt"
        self.latest = self.directory / "latest_student_yolo.pt"
        self.device = torch.device("cpu")
        self.model = BatchLossModel()
        self.logger = MagicMock()
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": {},
                "val_loss": 0.5,
            },
            self.checkpoint,
        )

    @patch("dataset_loader.DSECFusionDataset")
    def test_test_loader_uses_test_split_sequential_order_and_keeps_last_batch(
        self, dataset_class
    ):
        dataset_class.return_value = make_samples()
        loader = create_rgb_test_loader(
            Path("fake"), logger=self.logger, batch_size=4, num_workers=0
        )

        dataset_class.assert_called_once_with(
            dataset_path=Path("fake"),
            split="test",
            rgb_only=True,
            degradation="clean",
            degradation_strength=1.0,
            evaluation_seed=0,
        )
        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertIs(loader.collate_fn, yolo_collate_fn)
        self.assertFalse(loader.drop_last)
        self.assertFalse(loader.persistent_workers)
        batches = list(loader)
        self.assertEqual([len(images) for images, _ in batches], [4, 1])
        self.assertEqual(
            torch.cat([images[:, 0, 0, 0] for images, _ in batches]).tolist(),
            [0, 1, 2, 3, 4],
        )

    @patch("dataset_loader.DSECFusionDataset", return_value=make_samples())
    def test_test_loader_worker_configuration(self, dataset_class):
        loader = create_rgb_test_loader(Path("fake"), logger=self.logger, num_workers=2)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.num_workers, 2)
        self.assertTrue(loader.pin_memory)

    def test_invalid_test_loader_configuration_is_logged(self):
        for kwargs in ({"batch_size": 0}, {"batch_size": 33}, {"num_workers": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                create_rgb_test_loader(Path("fake"), logger=self.logger, **kwargs)
        self.assertEqual(self.logger.error.call_count, 3)

    @patch("dataset_loader.DSECFusionDataset", return_value=[])
    def test_empty_test_split_fails_clearly(self, dataset_class):
        with self.assertRaisesRegex(ValueError, "test split is empty"):
            create_rgb_test_loader(Path("fake"), logger=self.logger)
        self.logger.error.assert_called_once()

    def test_restores_best_weights_disables_gradients_and_counts_partial_batch(self):
        with torch.no_grad():
            self.model.weight.fill_(99.0)
        before = self.model.bn.running_mean.clone()

        results = evaluate_test_model(
            make_loader(),
            self.model,
            self.device,
            self.logger,
            self.checkpoint,
            compute_metrics=False,
        )

        self.assertEqual(self.model.weight.item(), 2.0)
        # Per-image loss, on the same scale as the training and validation loss.
        self.assertEqual(results["loss"], 3.0)
        self.assertEqual(
            self.model.observed,
            [
                (False, False, False, False, (4, 1), (4, 4), (4,)),
                (False, False, False, False, (1, 1), (1, 4), (1,)),
            ],
        )
        self.assertTrue(torch.equal(before, self.model.bn.running_mean))
        self.assertIsNone(self.model.weight.grad)
        self.logger.info.assert_any_call("Test samples: %d | Test batches: %d", 5, 2)
        self.assertAlmostEqual(results["box_loss"], 0.3, places=6)
        messages = [args[0] for args, _ in self.logger.info.call_args_list]
        self.assertTrue(any(message.startswith("Test loss") for message in messages))
        self.assertTrue(torch.is_grad_enabled())

    def test_batches_with_no_detection_targets_are_valid(self):
        result = evaluate_test_model(
            make_loader(empty_targets=True),
            self.model,
            self.device,
            self.logger,
            self.checkpoint,
            compute_metrics=False,
        )
        self.assertEqual(result["loss"], 3.0)
        self.assertEqual(self.model.observed[0][4:], ((0, 1), (0, 4), (0,)))

    def test_missing_checkpoint_does_not_use_current_weights(self):
        with self.assertRaisesRegex(FileNotFoundError, "Best checkpoint not found"):
            evaluate_test_model(
                make_loader(),
                self.model,
                self.device,
                self.logger,
                self.checkpoint.with_name("missing.pt"),
            )
        self.assertEqual(self.model.observed, [])

    def test_corrupt_checkpoint_is_rejected(self):
        self.checkpoint.write_bytes(b"this is not a torch checkpoint")
        with self.assertRaisesRegex(RuntimeError, "Could not load"):
            evaluate_test_model(
                make_loader(),
                self.model,
                self.device,
                self.logger,
                self.checkpoint,
            )
        self.assertEqual(self.model.observed, [])

    def test_incompatible_checkpoint_is_rejected(self):
        torch.save(
            {"model_state_dict": torch.nn.Linear(2, 2).state_dict()},
            self.checkpoint,
        )
        with self.assertRaisesRegex(RuntimeError, "Could not load"):
            evaluate_test_model(
                make_loader(),
                self.model,
                self.device,
                self.logger,
                self.checkpoint,
            )
        self.assertEqual(self.model.observed, [])

    def test_empty_test_loader_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "test loader is empty"):
            evaluate_test_model(
                make_loader(0),
                self.model,
                self.device,
                self.logger,
                self.checkpoint,
            )
        self.assertEqual(self.model.observed, [])

    def test_nonfinite_test_loss_does_not_report_success(self):
        nan_loss = (
            torch.tensor(float("nan")),
            {
                "box_loss": torch.tensor(0.0),
                "cls_loss": torch.tensor(0.0),
                "l1_loss": torch.tensor(0.0),
            },
        )
        with patch.object(self.model, "loss", return_value=nan_loss):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                evaluate_test_model(
                    make_loader(),
                    self.model,
                    self.device,
                    self.logger,
                    self.checkpoint,
                    compute_metrics=False,
                )
        messages = [args[0] for args, _ in self.logger.info.call_args_list]
        self.assertFalse(any(message.startswith("Test loss") for message in messages))

    def test_real_detector_reports_metrics_per_class_and_saves_plots(self):
        from utils import extract_trainable_model

        model = extract_trainable_model(ch=3, weights_path=None, device="cpu")
        torch.save({"model_state_dict": model.state_dict()}, self.checkpoint)
        samples = [
            (
                torch.rand(3, 64, 64),
                torch.tensor([[2.0, 0.5, 0.5, 0.3, 0.3], [0.0, 0.3, 0.6, 0.1, 0.2]]),
            )
            for _ in range(2)
        ]
        loader = DataLoader(samples, batch_size=2, collate_fn=yolo_collate_fn)
        save_dir = self.directory / "test"

        results = evaluate_test_model(
            loader, model, self.device, self.logger, self.checkpoint, save_dir=save_dir
        )

        for key in (
            "loss",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "agnostic_map50",
        ):
            self.assertIn(key, results)
        rows = results["per_class"]
        self.assertEqual([row["class"] for row in rows], ["all", "pedestrian", "car"])
        self.assertEqual(rows[0]["images"], 2)
        self.assertEqual(rows[0]["instances"], 4)
        self.assertEqual(rows[0]["map50"], results["map50"])
        logged = [args[0] for args, _ in self.logger.info.call_args_list]
        self.assertIn(format_results_table(rows)[0], logged)
        self.assertTrue(any(line.lstrip().startswith("all ") for line in logged))
        self.assertTrue((save_dir / "confusion_matrix.png").is_file())
        self.assertTrue((save_dir / "val_batch0_pred.jpg").is_file())

    def test_training_saves_best_validation_epoch_instead_of_last_epoch(self):
        self.checkpoint.unlink()
        loader = make_loader(4)
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with patch(
            "training.validate_model",
            side_effect=[val_results(1.0), val_results(3.0)],
        ):
            path = train_model(
                loader,
                loader,
                self.model,
                optimiser,
                self.device,
                self.logger,
                total_epochs=2,
                checkpoint_path=self.checkpoint,
                latest_path=self.latest,
            )

        saved = torch.load(path, weights_only=True)
        self.assertAlmostEqual(
            saved["model_state_dict"]["weight"].item(), 1.4, places=5
        )
        self.assertAlmostEqual(self.model.weight.item(), 0.8, places=5)

    def test_training_improvement_overwrites_previous_best(self):
        self.checkpoint.unlink()
        loader = make_loader(4)
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with patch(
            "training.validate_model",
            side_effect=[val_results(3.0), val_results(1.0)],
        ):
            train_model(
                loader,
                loader,
                self.model,
                optimiser,
                self.device,
                self.logger,
                total_epochs=2,
                checkpoint_path=self.checkpoint,
                latest_path=self.latest,
            )

        saved = torch.load(self.checkpoint, weights_only=True)
        self.assertAlmostEqual(
            saved["model_state_dict"]["weight"].item(), 0.8, places=5
        )

    def test_invalid_validation_loss_preserves_old_checkpoint(self):
        loader = make_loader(4)
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with patch(
            "training.validate_model",
            return_value=val_results(float("nan")),
        ):
            with self.assertRaisesRegex(ValueError, "validation loss"):
                train_model(
                    loader,
                    loader,
                    self.model,
                    optimiser,
                    self.device,
                    self.logger,
                    total_epochs=1,
                    checkpoint_path=self.checkpoint,
                    latest_path=self.latest,
                )
        self.assertEqual(
            torch.load(self.checkpoint, weights_only=True)["model_state_dict"][
                "weight"
            ].item(),
            2.0,
        )

    def _run_main(self, *extra_args, role="student", rgb_only=True):
        events = []
        self.calls = {}
        loader = make_loader()
        logger = MagicMock()
        with (
            unittest.mock.patch.multiple(
                training,
                select_device=MagicMock(return_value=self.device),
                create_dataloaders=MagicMock(
                    side_effect=lambda **kwargs: (
                        events.append("dataloaders")
                        or self.calls.update(dataloaders=kwargs)
                        or (loader, loader)
                    )
                ),
                extract_trainable_model=MagicMock(
                    side_effect=lambda **kwargs: (
                        self.calls.update(model=kwargs) or self.model
                    )
                ),
                create_optimiser=MagicMock(return_value=MagicMock()),
                train_model=MagicMock(
                    side_effect=lambda *args, **kwargs: (
                        events.append("train")
                        or self.calls.update(train=kwargs)
                        or self.checkpoint
                    )
                ),
                create_test_loader=MagicMock(
                    side_effect=lambda **kwargs: (
                        events.append("test_loader")
                        or self.calls.update(test_loader=kwargs)
                        or loader
                    )
                ),
                evaluate_test_model=MagicMock(
                    side_effect=lambda **kwargs: (
                        events.append("test_evaluation")
                        or self.calls.update(test=kwargs)
                    )
                ),
            ),
            patch.object(
                training.log, "create_log_directory", return_value=self.directory
            ),
            patch.object(training.log, "create_logger", return_value=logger),
            patch.object(training.log, "log_command"),
            patch.object(training.log, "log_model_config"),
            patch(
                "sys.argv",
                ["train_rgb_student.py", "--dataset-path", "fake", *extra_args],
            ),
        ):
            training.main(role=role, rgb_only=rgb_only)

        return events

    def test_main_runs_test_evaluation_after_training(self):
        self.assertEqual(
            self._run_main(),
            ["dataloaders", "train", "test_loader", "test_evaluation"],
        )
        self.assertEqual(self.calls["test"]["checkpoint_path"], self.checkpoint)
        self.assertEqual(self.calls["test"]["save_dir"], self.directory / "test")

    def test_main_skip_test_does_not_open_test_data(self):
        self.assertEqual(self._run_main("--skip-test"), ["dataloaders", "train"])

    def test_teacher_runs_the_same_pipeline_with_event_channels(self):
        student_events = self._run_main()
        student_calls = self.calls
        teacher_events = self._run_main(role="teacher", rgb_only=False)
        teacher_calls = self.calls

        self.assertEqual(teacher_events, student_events)
        self.assertEqual(
            (student_calls["model"]["ch"], teacher_calls["model"]["ch"]), (3, 5)
        )
        for step in ("dataloaders", "test_loader"):
            self.assertTrue(student_calls[step]["rgb_only"])
            self.assertFalse(teacher_calls[step]["rgb_only"])
        self.assertEqual(
            teacher_calls["train"]["checkpoint_path"].name, "best_teacher_yolo.pt"
        )


if __name__ == "__main__":
    unittest.main()
