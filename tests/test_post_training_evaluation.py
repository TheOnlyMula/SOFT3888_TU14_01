import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, SequentialSampler

from dataset_loader import yolo_rgb_collate_fn
from train_rgb_student import create_rgb_test_loader, evaluate_test_model, train_model
import train_rgb_student as training


class BatchLossModel(torch.nn.Module):
    """Small real torch module; deliberately returns a batch-scaled loss."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.bn = torch.nn.BatchNorm2d(3)
        self.dropout = torch.nn.Dropout()
        self.observed = []

    def forward(self, batch):
        self.dropout(self.bn(batch["img"]))
        self.observed.append((
            self.training, self.bn.training, self.dropout.training,
            torch.is_grad_enabled(), batch["cls"].shape,
            batch["bboxes"].shape, batch["batch_idx"].shape,
        ))
        count = batch["img"].size(0)
        return torch.stack((self.weight * count, self.weight * count / 2)), None


def make_samples(count=5, empty_targets=False):
    return [
        (torch.ones(3, 8, 8) * index,
         torch.empty((0, 5)) if empty_targets else
         torch.tensor([[1.0, 0.5, 0.5, 0.2, 0.2]]))
        for index in range(count)
    ]


def make_loader(count=5, empty_targets=False):
    return DataLoader(make_samples(count, empty_targets), batch_size=4,
                      collate_fn=yolo_rgb_collate_fn, shuffle=False)


class PostTrainingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.checkpoint = Path(self.temp_dir.name) / "best_student_yolo.pt"
        self.device = torch.device("cpu")
        self.model = BatchLossModel()
        torch.save(self.model.state_dict(), self.checkpoint)

    @patch("train_rgb_student.DSECFusionDataset")
    def test_test_loader_uses_test_split_sequential_order_and_keeps_last_batch(self, dataset_class):
        dataset_class.return_value = make_samples()
        loader = create_rgb_test_loader(Path("fake"), batch_size=4, num_workers=0)
        dataset_class.assert_called_once_with(
            dataset_path=Path("fake"), split="test", rgb_only=True)
        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertIs(loader.collate_fn, yolo_rgb_collate_fn)
        self.assertFalse(loader.drop_last)
        self.assertFalse(loader.persistent_workers)
        batches = list(loader)
        self.assertEqual([len(images) for images, _ in batches], [4, 1])
        self.assertEqual(torch.cat([images[:, 0, 0, 0] for images, _ in batches]).tolist(),
                         [0, 1, 2, 3, 4])

    @patch("train_rgb_student.DSECFusionDataset", return_value=make_samples())
    def test_worker_configuration(self, dataset_class):
        loader = create_rgb_test_loader(Path("fake"), num_workers=2)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.num_workers, 2)
        self.assertTrue(loader.pin_memory)

    def test_invalid_test_loader_configuration(self):
        for kwargs in ({"batch_size": 0}, {"batch_size": 51}, {"num_workers": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                create_rgb_test_loader(Path("fake"), **kwargs)

    @patch("train_rgb_student.DSECFusionDataset", return_value=[])
    def test_empty_test_split_fails_clearly(self, dataset_class):
        with self.assertRaisesRegex(ValueError, "test split is empty"):
            create_rgb_test_loader(Path("fake"))

    def test_restores_best_weights_disables_gradients_and_counts_partial_batch(self):
        with torch.no_grad():
            self.model.weight.fill_(99.0)
        before = self.model.bn.running_mean.clone()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            batch_loss, image_loss = evaluate_test_model(
                make_loader(), self.model, self.device, self.checkpoint)
        self.assertEqual(self.model.weight.item(), 2.0)
        # Batch losses: 12 for four images, 3 for one image. Total = 15.
        self.assertEqual(batch_loss, 7.5)
        self.assertEqual(image_loss, 3.0)
        self.assertEqual(self.model.observed, [
            (False, False, False, False, (4, 1), (4, 4), (4,)),
            (False, False, False, False, (1, 1), (1, 4), (1,)),
        ])
        self.assertTrue(torch.equal(before, self.model.bn.running_mean))
        self.assertIsNone(self.model.weight.grad)
        self.assertIn("Average test batch loss: 7.5000", output.getvalue())
        self.assertIn("Average test per-image loss: 3.0000", output.getvalue())
        self.assertTrue(torch.is_grad_enabled())

    def test_batches_with_no_detection_targets_are_valid(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_test_model(make_loader(empty_targets=True), self.model,
                                         self.device, self.checkpoint)
        self.assertEqual(result, (7.5, 3.0))
        self.assertEqual(self.model.observed[0][4:], ((0, 1), (0, 4), (0,)))

    def test_missing_checkpoint_does_not_use_current_weights(self):
        with self.assertRaisesRegex(FileNotFoundError, "Best checkpoint not found"):
            evaluate_test_model(make_loader(), self.model, self.device,
                                self.checkpoint.with_name("missing.pt"))
        self.assertEqual(self.model.observed, [])

    def test_corrupt_checkpoint_does_not_print_success_summary(self):
        self.checkpoint.write_bytes(b"this is not a torch checkpoint")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(RuntimeError):
            evaluate_test_model(make_loader(), self.model, self.device, self.checkpoint)
        self.assertEqual(self.model.observed, [])
        self.assertNotIn("Average test", output.getvalue())

    def test_incompatible_checkpoint_is_rejected(self):
        torch.save(torch.nn.Linear(2, 2).state_dict(), self.checkpoint)
        with self.assertRaisesRegex(RuntimeError, "Could not load"):
            evaluate_test_model(make_loader(), self.model, self.device, self.checkpoint)
        self.assertEqual(self.model.observed, [])

    def test_empty_test_loader_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "test loader is empty"):
            evaluate_test_model(make_loader(0), self.model, self.device, self.checkpoint)
        self.assertEqual(self.model.observed, [])

    def test_nonfinite_test_loss_does_not_print_success_summary(self):
        output = io.StringIO()
        with patch.object(self.model, "forward", return_value=(torch.tensor(float("nan")), None)):
            with contextlib.redirect_stdout(output), self.assertRaisesRegex(ValueError, "Non-finite"):
                evaluate_test_model(make_loader(), self.model, self.device, self.checkpoint)
        self.assertNotIn("Average test", output.getvalue())

    def test_training_saves_best_validation_epoch_instead_of_last_epoch(self):
        loader = make_loader(4)
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with patch("train_rgb_student.validate_model", side_effect=[1.0, 3.0]):
            with contextlib.redirect_stdout(io.StringIO()):
                path = train_model(loader, loader, self.model, optimiser,
                                   self.device, self.checkpoint)
        saved = torch.load(path, weights_only=True)
        self.assertAlmostEqual(saved["weight"].item(), 1.4, places=5)
        self.assertAlmostEqual(self.model.weight.item(), 0.8, places=5)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate_test_model(loader, self.model, self.device, path)
        self.assertAlmostEqual(self.model.weight.item(), 1.4, places=5)

    def test_training_improvement_overwrites_previous_best(self):
        loader = make_loader(4)
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with patch("train_rgb_student.validate_model", side_effect=[3.0, 1.0]):
            with contextlib.redirect_stdout(io.StringIO()):
                train_model(loader, loader, self.model, optimiser, self.device, self.checkpoint)
        saved = torch.load(self.checkpoint, weights_only=True)
        self.assertAlmostEqual(saved["weight"].item(), 0.8, places=5)

    def test_invalid_validation_loss_stops_run_without_returning_old_checkpoint(self):
        loader = make_loader(4)
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with patch("train_rgb_student.validate_model", return_value=float("nan")):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "validation loss"):
                train_model(loader, loader, self.model, optimiser, self.device, self.checkpoint)
        # The old checkpoint exists, but the failed run cannot report it as a new best.
        self.assertEqual(torch.load(self.checkpoint, weights_only=True)["weight"].item(), 2.0)

    def _run_main(self, *extra_args):
        events = []
        loader = make_loader()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["train_rgb_student.py", "--dataset-path", "fake", *extra_args]))
            stack.enter_context(patch.object(training, "select_device", return_value=self.device))
            stack.enter_context(patch.object(training, "create_rgb_dataloaders", return_value=(loader, loader)))
            stack.enter_context(patch.object(training, "extract_trainable_model", return_value=self.model))
            stack.enter_context(patch.object(training, "create_optimiser"))
            stack.enter_context(patch.object(training, "train_model", side_effect=lambda *a: events.append("train") or self.checkpoint))
            stack.enter_context(patch.object(training, "create_rgb_test_loader", side_effect=lambda **kw: events.append("test_loader") or loader))
            evaluation = stack.enter_context(patch.object(training, "evaluate_test_model", side_effect=lambda *a: events.append("evaluate")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            training.main()
        return events, loader, evaluation

    def test_main_runs_test_only_after_training_returns_checkpoint(self):
        events, loader, evaluation = self._run_main()
        self.assertEqual(events, ["train", "test_loader", "evaluate"])
        evaluation.assert_called_once_with(loader, self.model, self.device, self.checkpoint)


    def test_main_skip_test_keeps_training_without_opening_test_data(self):
        events, _, evaluation = self._run_main("--skip-test")
        self.assertEqual(events, ["train"])
        evaluation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
