from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import torch
import yaml
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


def val_results(loss, map50_95=0.0):
    return {
        "loss": loss,
        "box_loss": 0.0,
        "cls_loss": 0.0,
        "l1_loss": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map50_95": map50_95,
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

    def test_training_saves_highest_map_epoch_instead_of_last_epoch(self):
        self.checkpoint.unlink()
        loader = make_loader(4)
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
        validated_weights = []
        # Epoch 1 has the higher loss but the higher mAP50-95, so selection must follow mAP
        validation_results = iter([val_results(3.0, map50_95=0.5), val_results(1.0, map50_95=0.2)])

        def validate_and_record(val_loader, model, device, logger):
            self.assertIs(val_loader, loader)
            self.assertIs(device, self.device)
            self.assertIs(logger, self.logger)
            validated_weights.append(model.weight.detach().clone())
            return next(validation_results)

        with patch("training.validate_model", side_effect=validate_and_record):
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

        # model_state_dict holds EMA weights, so check the selected epoch rather than raw weights.
        saved = torch.load(path, weights_only=True)
        self.assertEqual(saved["epoch"], 1)
        self.assertEqual(saved["val_metrics"]["map50_95"], 0.5)
        # The saved weights are exactly the ones validated in the best epoch
        torch.testing.assert_close(saved["model_state_dict"]["weight"], validated_weights[0])
        # The in-memory model keeps training past the best epoch: it matches the latest checkpoint.
        latest = torch.load(self.latest, weights_only=True)
        self.assertEqual(latest["epoch"], 2)
        torch.testing.assert_close(
            latest["raw_model_state_dict"]["weight"], self.model.weight.detach()
        )

    def _train_one_epoch_recording_steps(self, batches, **kwargs):
        loader = make_loader(4 * batches)
        step_lrs = []

        class RecordingSGD(torch.optim.SGD):
            def step(self, closure=None):
                step_lrs.append(self.param_groups[0]["lr"])
                return super().step(closure)

        optimiser = RecordingSGD(self.model.parameters(), lr=0.1)
        self.checkpoint.unlink()
        with patch("training.validate_model", return_value=val_results(1.0, map50_95=0.5)):
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
                **kwargs,
            )
        return step_lrs

    def test_gradients_accumulate_to_the_nominal_batch_size(self):
        # 2 epochs of 3 batches (4 images each) with a nominal batch of 8: like Ultralytics, a
        # step happens every 2 batches and the leftover batch carries over into the next epoch.
        step_lrs = self._train_one_epoch_recording_steps(
            batches=3, nominal_batch_size=8, warmup_epochs=0
        )
        self.assertEqual(len(step_lrs), 3)

    def test_learning_rate_warms_up_from_zero_then_follows_the_schedule(self):
        step_lrs = self._train_one_epoch_recording_steps(
            batches=4, nominal_batch_size=4, warmup_epochs=1
        )
        # Warmup ramps from 0 towards the first epoch's rate over the 4 warmup batches
        for actual, expected in zip(step_lrs[:4], [0.0, 0.025, 0.05, 0.075]):
            self.assertAlmostEqual(actual, expected, places=7)
        # Then the linear schedule alone sets the rate for the second of 2 epochs
        for actual in step_lrs[4:]:
            self.assertAlmostEqual(actual, 0.1 * training.lr_factor(1, 2), places=7)
        self.assertAlmostEqual(training.lr_factor(1, 2), 0.505)

    def test_warmup_never_covers_the_final_epoch(self):
        # A 3-epoch warmup in a 2-epoch run is cut to 1 epoch, as in Ultralytics
        step_lrs = self._train_one_epoch_recording_steps(
            batches=4, nominal_batch_size=4, warmup_epochs=3
        )
        self.assertAlmostEqual(step_lrs[3], 0.075, places=7)
        self.assertAlmostEqual(step_lrs[4], 0.1 * training.lr_factor(1, 2), places=7)

    def test_training_map_improvement_overwrites_previous_best(self):
        self.checkpoint.unlink()
        loader = make_loader(4)
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
        validated_weights = []
        # Epoch 2 has the higher loss but the higher mAP50-95, so it must replace epoch 1
        validation_results = iter([val_results(1.0, map50_95=0.2), val_results(3.0, map50_95=0.5)])

        def validate_and_record(val_loader, model, device, logger):
            self.assertIs(val_loader, loader)
            self.assertIs(device, self.device)
            self.assertIs(logger, self.logger)
            validated_weights.append(model.weight.detach().clone())
            return next(validation_results)

        with patch("training.validate_model", side_effect=validate_and_record):
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
        self.assertEqual(saved["epoch"], 2)
        self.assertEqual(saved["val_metrics"]["map50_95"], 0.5)
        torch.testing.assert_close(saved["model_state_dict"]["weight"], validated_weights[1])

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

    def _run_main(self, *extra_args, role="student", rgb_only=True, argv=None):
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
            ) as self.create_log_directory,
            patch.object(training.log, "create_logger", return_value=logger),
            patch.object(training.log, "log_command") as self.log_command,
            patch.object(training.log, "log_model_config"),
            patch(
                "sys.argv",
                argv or ["train_rgb_student.py", "--dataset-path", "fake", *extra_args],
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

    def _save_original_command(self, *args, script="train_rgb_student.py"):
        (self.directory / "run_args.yaml").write_text(yaml.safe_dump([script, *args]))
        (self.directory / "latest_student_yolo.pt").touch()

    def test_resume_continues_in_the_original_folder_with_its_settings(self):
        self._save_original_command("--dataset-path", "fake", "--epochs", "7", "--batch-size", "2")
        self._run_main(
            argv=["train_rgb_student.py", "--resume", str(self.directory), "--num-workers", "3"]
        )

        self.create_log_directory.assert_not_called()
        self.log_command.assert_not_called()  # the original command files are kept
        self.assertEqual(self.calls["train"]["latest_path"], self.directory / "latest_student_yolo.pt")
        self.assertEqual(self.calls["train"]["total_epochs"], 7)
        self.assertEqual(self.calls["model"]["epochs"], 7)
        self.assertEqual(self.calls["dataloaders"]["batch_size"], 2)
        self.assertEqual(self.calls["dataloaders"]["num_workers"], 3)  # allowed override

    def test_resume_without_a_latest_checkpoint_fails(self):
        self._save_original_command("--dataset-path", "fake")
        (self.directory / "latest_student_yolo.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "no latest checkpoint"):
            self._run_main(argv=["train_rgb_student.py", "--resume", str(self.directory)])

    def test_resume_refuses_to_change_training_settings(self):
        self._save_original_command("--dataset-path", "fake", "--epochs", "7")
        argv = ["train_rgb_student.py", "--resume", str(self.directory), "--epochs", "9"]
        with patch("sys.argv", argv), self.assertRaises(SystemExit):
            training.parse_args("test")

    def test_resume_refuses_a_run_from_another_script(self):
        self._save_original_command("--dataset-path", "fake", script="train_distilled_student.py")
        argv = ["train_rgb_student.py", "--resume", str(self.directory)]
        with patch("sys.argv", argv), self.assertRaises(SystemExit):
            training.parse_args("test")

    def test_resume_reads_older_runs_from_run_command_txt(self):
        (self.directory / "run_command.txt").write_text(
            "train_rgb_student.py --dataset-path dataset --epochs 100 --batch-size 6 --num-workers 4"
        )
        with patch("sys.argv", ["train_rgb_student.py", "--resume", str(self.directory)]):
            args = training.parse_args("test")
        self.assertEqual(args.dataset_path, Path("dataset"))
        self.assertEqual((args.epochs, args.batch_size, args.num_workers), (100, 6, 4))
        self.assertEqual(args.resume, self.directory)

    def test_new_runs_need_a_dataset_path(self):
        with patch("sys.argv", ["train_rgb_student.py"]), self.assertRaises(SystemExit):
            training.parse_args("test")

    def test_interrupted_training_resumes_after_the_last_saved_epoch(self):
        self.checkpoint.unlink()
        loader = make_loader(4)
        settings = dict(total_epochs=3, checkpoint_path=self.checkpoint, latest_path=self.latest)

        # Epoch 1 finishes and is saved; the run dies while validating epoch 2
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with patch(
            "training.validate_model",
            side_effect=[val_results(1.0, map50_95=0.5), RuntimeError("power cut")],
        ):
            with self.assertRaises(RuntimeError):
                train_model(loader, loader, self.model, optimiser, self.device, self.logger, **settings)
        self.assertEqual(torch.load(self.latest, weights_only=True)["epoch"], 1)

        # A fresh process resumes: only epochs 2 and 3 run, and the best is still chosen by mAP
        model = BatchLossModel()
        optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
        with patch(
            "training.validate_model",
            side_effect=[val_results(1.0, map50_95=0.4), val_results(1.0, map50_95=0.6)],
        ) as validate:
            train_model(loader, loader, model, optimiser, self.device, self.logger, **settings)

        self.assertEqual(validate.call_count, 2)
        self.assertEqual(torch.load(self.latest, weights_only=True)["epoch"], 3)
        best = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual((best["epoch"], best["val_metrics"]["map50_95"]), (3, 0.6))

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
