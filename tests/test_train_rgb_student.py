import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from loaders.dataset_loader import limit_dataset, yolo_collate_fn
from utils.model import DSEC_NAMES, build_model, set_loss_schedule
from loaders.dataset_loader import create_rgb_dataloaders
from degradation.conditions import DEFAULT_AUGMENTATION_SEED, DEFAULT_STRENGTH_RANGE
from evaluation.metrics import validate_model
from ultralytics.optim import MuSGD
from utils.model import (
    OPTIMISER_LR,
    create_optimiser,
    extract_trainable_model,
    best_checkpoint_map50_95,
    load_checkpoint,
    select_device,
)


def param_ids(group):
    return {id(p) for p in group["params"]}


def non_empty(optimiser):
    # Like Ultralytics, every group is split into a 3x and a 1x half, even when one is empty
    return [g for g in optimiser.param_groups if g["params"]]


def groups_named(optimiser, name):
    """Both halves (3x and 1x learning rate) of the optimiser groups with this `param_group` name."""
    return [g for g in optimiser.param_groups if g["param_group"] == name]


def ids_in(groups):
    return set().union(*(param_ids(g) for g in groups))


def group_named(optimiser, name):
    """The non-boosted optimiser group with this Ultralytics `param_group` name."""
    return next(
        g for g in optimiser.param_groups
        if g["param_group"] == name and g["lr"] == OPTIMISER_LR
    )


class TrainRgbStudentTests(unittest.TestCase):
    def test_select_device(self):
        with patch("utils.model.torch.cuda.is_available", return_value=True):
            self.assertEqual(select_device().type, "cuda")
        with patch("utils.model.torch.cuda.is_available", return_value=False):
            self.assertEqual(select_device().type, "cpu")

    @patch("loaders.dataset_loader.DSECFusionDataset")
    def test_dataloader_configuration(self, dataset_class):
        dataset_class.side_effect = [[None] * 8, [None] * 2]
        logger = MagicMock()

        train_loader, val_loader = create_rgb_dataloaders(
            dataset_path=Path("fake-dataset"),
            logger=logger,
            batch_size=4,
            num_workers=4,
        )

        self.assertEqual(len(train_loader.dataset), 8)
        self.assertEqual(len(val_loader.dataset), 2)
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)
        self.assertEqual(train_loader.batch_size, 4)
        self.assertEqual(val_loader.batch_size, 4)
        self.assertEqual(train_loader.num_workers, 4)
        self.assertEqual(val_loader.num_workers, 4)
        self.assertTrue(train_loader.pin_memory)
        self.assertTrue(val_loader.pin_memory)
        self.assertTrue(train_loader.persistent_workers)
        self.assertTrue(val_loader.persistent_workers)
        self.assertIs(train_loader.collate_fn, yolo_collate_fn)
        self.assertIs(val_loader.collate_fn, yolo_collate_fn)
        self.assertEqual(
            dataset_class.call_args_list,
            [
                call(
                    dataset_path=Path("fake-dataset"),
                    split="train",
                    rgb_only=True,
                    augment=False,
                    with_event_window=False,
                    rgb_dropout=0.0,
                    augmentation_shares=None,
                    dark_strength_range=DEFAULT_STRENGTH_RANGE,
                    overexposed_strength_range=DEFAULT_STRENGTH_RANGE,
                    augmentation_seed=DEFAULT_AUGMENTATION_SEED,
                ),
                call(
                    dataset_path=Path("fake-dataset"),
                    split="val",
                    rgb_only=True,
                    degradation="clean",
                    degradation_strength=1.0,
                    evaluation_seed=0,
                ),
            ],
        )

    def test_limit_dataset_spreads_samples_across_the_split(self):
        dataset = list(range(100))
        self.assertIs(limit_dataset(dataset, None), dataset)
        self.assertIs(limit_dataset(dataset, 500), dataset)
        self.assertEqual(limit_dataset(dataset, 4).indices, [0, 25, 50, 75])
        with self.assertRaises(ValueError):
            limit_dataset(dataset, 0)

    def test_invalid_dataloader_configuration_is_logged(self):
        logger = MagicMock()
        for kwargs in ({"batch_size": 0}, {"batch_size": 33}, {"num_workers": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                create_rgb_dataloaders(Path("fake"), logger=logger, **kwargs)
        self.assertEqual(logger.error.call_count, 3)

    @patch("utils.model.build_model")
    def test_extract_trainable_model(self, mock_build_model):
        model = torch.nn.Linear(2, 2).eval()
        mock_build_model.return_value = model

        result = extract_trainable_model(
            ch=3,
            weights_path="dummy.pt",
            device=torch.device("cpu"),
            epochs=50,
        )

        mock_build_model.assert_called_once_with(3, "dummy.pt")
        self.assertIs(result, model)
        self.assertTrue(result.training)
        # Ultralytics' default loss gains, as its own trainer sets them, with the run's epochs
        self.assertEqual((result.args.box, result.args.cls, result.args.dfl), (7.5, 0.5, 1.5))
        self.assertEqual(result.args.epochs, 50)

    def test_build_model_matches_dsec_setup(self):
        for ch in (3, 5):
            with self.subTest(ch=ch):
                model = build_model(ch=ch)
                self.assertEqual(model.model[0].conv.in_channels, ch)
                self.assertEqual(model.model[-1].nc, 8)
                self.assertEqual(model.names, DSEC_NAMES)

    def test_teacher_starts_as_the_pretrained_rgb_detector(self):
        student = build_model(ch=3, weights_path="yolo26n.pt").eval()
        teacher = build_model(ch=5, weights_path="yolo26n.pt").eval()
        rgb, events = torch.rand(2, 3, 32, 32), torch.rand(2, 2, 32, 32)
        with torch.no_grad():
            # Event filters start at zero, so the first layer sees only the pretrained RGB filters
            torch.testing.assert_close(
                teacher.model[0](torch.cat((rgb, events), dim=1)), student.model[0](rgb), rtol=0, atol=0
            )
        self.assertTrue(teacher.model[0].conv.weight[:, 3:].eq(0).all())
        teacher.train()
        teacher.model[0](torch.cat((rgb, events), dim=1)).sum().backward()
        # ...but they still learn: the events reach them through the gradient
        self.assertGreater(teacher.model[0].conv.weight.grad[:, 3:].abs().sum(), 0)

    def test_optimiser_configuration(self):
        model = build_model(ch=3)
        optimiser = create_optimiser(model)

        self.assertIsInstance(optimiser, MuSGD)
        for group in non_empty(optimiser):
            self.assertEqual(group["momentum"], 0.9)
            self.assertTrue(group["nesterov"])

        # Conv and linear weights get the Muon update with weight decay; biases plain SGD without
        parameters = list(model.parameters())
        muon = groups_named(optimiser, "muon")
        bias = groups_named(optimiser, "bias")
        self.assertEqual(ids_in(muon), {id(p) for p in parameters if p.ndim in (2, 4)})
        self.assertTrue(all(g["use_muon"] and g["weight_decay"] == 0.0005 for g in muon))
        self.assertEqual(ids_in(bias), {id(p) for name, p in model.named_parameters() if name.endswith(".bias")})
        self.assertTrue(all(not g["use_muon"] and g["weight_decay"] == 0.0 for g in bias))

        # Every parameter is optimised exactly once
        optimised = [id(p) for g in optimiser.param_groups for p in g["params"]]
        self.assertEqual(sorted(optimised), sorted(id(p) for p in parameters))

    def test_normalisation_weights_get_no_weight_decay(self):
        model = build_model(ch=3)
        optimiser = create_optimiser(model)
        norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        norm_weights = groups_named(optimiser, "bn")
        self.assertEqual(ids_in(norm_weights), {id(m.weight) for m in norms})
        self.assertTrue(all(g["weight_decay"] == 0.0 for g in norm_weights))
        self.assertTrue({id(m.bias) for m in norms} <= ids_in(groups_named(optimiser, "bias")))

    def test_yolo26_classification_head_trains_at_three_times_the_rate(self):
        model = build_model(ch=3)
        optimiser = create_optimiser(model, batch_size=6)
        head = model.model[-1]
        boosted = {id(p) for layer in (head.cv3, head.one2one_cv3) for p in layer.parameters()}
        for group in non_empty(optimiser):
            expected_lr = 3 * OPTIMISER_LR if param_ids(group) & boosted else OPTIMISER_LR
            self.assertEqual(group["lr"], expected_lr)
        self.assertEqual(
            set().union(*(param_ids(g) for g in optimiser.param_groups if g["lr"] > OPTIMISER_LR)),
            boosted,
        )
        # Weight decay scales with the images per step: 6 images x 11 batches against 64
        self.assertAlmostEqual(group_named(optimiser, "muon")["weight_decay"], 0.0005 * 66 / 64)

    def test_checkpoint_restores_model_optimiser_ema_and_schedule(self):
        from torch.optim.lr_scheduler import LambdaLR
        from ultralytics.utils.torch_utils import ModelEMA

        def training_state():
            model = torch.nn.Linear(10, 2)
            optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)
            return model, optimiser, ModelEMA(model), LambdaLR(optimiser, lambda epoch: 0.5 ** epoch)

        original_model, original_optimiser, original_ema, original_scheduler = training_state()
        torch.nn.functional.mse_loss(original_model(torch.randn(4, 10)), torch.randn(4, 2)).backward()
        original_optimiser.step()
        original_ema.update(original_model)
        original_scheduler.step()
        checkpoint = {
            "epoch": 5,
            "model_state_dict": original_ema.ema.state_dict(),
            "raw_model_state_dict": original_model.state_dict(),
            "ema_updates": original_ema.updates,
            "optimiser_state_dict": original_optimiser.state_dict(),
            "scheduler_state_dict": original_scheduler.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_student_yolo.pt"
            torch.save(checkpoint, path)
            model, optimiser, ema, scheduler = training_state()
            loaded = load_checkpoint(path, torch.device("cpu"), model, optimiser, ema, scheduler)

        self.assertEqual(loaded, 5)
        for original, restored in zip(original_model.parameters(), model.parameters()):
            self.assertTrue(torch.equal(original, restored))
        for original, restored in zip(original_ema.ema.parameters(), ema.ema.parameters()):
            self.assertTrue(torch.equal(original, restored))
        self.assertEqual(ema.updates, original_ema.updates)
        self.assertEqual(optimiser.state_dict()["param_groups"], original_optimiser.state_dict()["param_groups"])
        self.assertEqual(scheduler.last_epoch, 1)

    def test_missing_checkpoint_starts_from_epoch_zero(self):
        model = torch.nn.Linear(10, 2)
        optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)
        result = load_checkpoint("this_file_does_not_exist.pt", torch.device("cpu"), model, optimiser, None, None)
        self.assertEqual(result, 0)

    def test_best_checkpoint_map_is_read_or_minus_infinity_before_one_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best_student_yolo.pt"
            self.assertEqual(best_checkpoint_map50_95(path, torch.device("cpu")), float("-inf"))
            torch.save({"epoch": 3, "val_metrics": {"map50_95": 0.4}}, path)
            self.assertEqual(best_checkpoint_map50_95(path, torch.device("cpu")), 0.4)

    def test_validation_uses_eval_mode_and_disables_gradients(self):
        model = MagicMock()
        logger = MagicMock()
        images = torch.randn(2, 3, 32, 32)
        targets = torch.tensor(
            [
                [0, 1, 0.5, 0.5, 0.2, 0.2],
                [1, 2, 0.4, 0.4, 0.1, 0.1],
            ],
            dtype=torch.float32,
        )
        gradient_states = []

        def loss_call(batch, preds):
            gradient_states.append(torch.is_grad_enabled())
            return torch.tensor([1.5]), {
                "box_loss": torch.tensor(0.5),
                "cls_loss": torch.tensor(0.4),
                "l1_loss": torch.tensor(0.3),
            }

        model.loss.side_effect = loss_call
        losses = validate_model(
            [(images, targets)],
            model,
            torch.device("cpu"),
            logger,
            compute_metrics=False,
        )

        model.eval.assert_called_once()
        self.assertEqual(gradient_states, [False])
        # Loss is reported per image so it matches the training loss scale.
        self.assertEqual(losses["loss"], 0.75)
        self.assertAlmostEqual(losses["box_loss"], 0.5)
        self.assertAlmostEqual(losses["cls_loss"], 0.4, places=6)
        self.assertAlmostEqual(losses["l1_loss"], 0.3, places=6)

    def test_loss_schedule_moves_weight_to_one_to_one_head(self):
        model = extract_trainable_model(
            ch=3, weights_path=None, device="cpu", epochs=10
        )
        set_loss_schedule(model, 0)
        self.assertAlmostEqual(model.criterion.o2m, 0.8)
        set_loss_schedule(model, 9)
        self.assertAlmostEqual(model.criterion.o2m, 0.1)
        self.assertAlmostEqual(model.criterion.o2o, 0.9)


if __name__ == "__main__":
    unittest.main()


class EpochResultsTests(unittest.TestCase):
    def test_results_csv_replaces_a_repeated_epoch_and_draws_the_curves(self):
        import csv
        from utils import log

        def row(epoch, loss):
            return {"epoch": epoch, "train/loss": loss, "val/loss": loss,
                    "metrics/mAP50": 0.1 * epoch, "metrics/mAP50-95": 0.05 * epoch, "lr": 0.01}

        with tempfile.TemporaryDirectory() as directory:
            for epoch, loss in ((1, 3.0), (2, 2.0), (3, 1.5)):
                log.log_epoch_results(row(epoch, loss), Path(directory), MagicMock())
            # A resumed run repeats epoch 3 after an interruption
            log.log_epoch_results(row(3, 1.0), Path(directory), MagicMock())

            with open(Path(directory) / "results.csv", newline="") as file:
                saved = list(csv.DictReader(file))
            self.assertEqual([r["epoch"] for r in saved], ["1", "2", "3"])
            self.assertEqual(float(saved[-1]["train/loss"]), 1.0)
            self.assertTrue((Path(directory) / "results.png").is_file())

    def test_a_distillation_run_draws_its_curves(self):
        from training.reporting import results_row
        from utils import log

        val = {"loss": 2.0, "box_loss": 1.0, "cls_loss": 1.0, "l1_loss": 0.0,
               "precision": 0.5, "recall": 0.4, "map50": 0.4, "map50_95": 0.25}
        weights = {"mean": 0.5, "min": 0.0, "max": 1.0, "zero_fraction": 0.1}
        with tempfile.TemporaryDirectory() as directory, \
                patch("ultralytics.utils.plotting.LOGGER") as ultralytics_logger:
            for epoch in (1, 2):
                train = {"loss": 3.0 / epoch, "box_loss": 1.5, "cls_loss": 1.5, "l1_loss": 0.0, "distill_loss": 0.06}
                log.log_epoch_results(results_row(epoch, train, val, weights, 0.01), Path(directory), MagicMock())
            self.assertTrue((Path(directory) / "results.png").is_file())
        ultralytics_logger.error.assert_not_called()

    def test_a_plot_that_cannot_be_saved_does_not_stop_training(self):
        # Windows refuses to overwrite results.png while it is open in a viewer
        import csv
        from utils import log

        logger = MagicMock()
        with tempfile.TemporaryDirectory() as directory,                 patch("utils.log.plot_results", side_effect=OSError(22, "Invalid argument")):
            log.log_epoch_results({"epoch": 1, "train/loss": 2.0}, Path(directory), logger)
            with open(Path(directory) / "results.csv", newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)
        logger.warning.assert_called_once()


class RunFolderAndLoggerTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)

    def test_run_folders_are_numbered_per_role_and_never_reused(self):
        import os
        from utils import log

        # create_log_directory works under runs/ in the current directory
        previous = os.getcwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, previous)
        (self.directory / "runs" / "student" / "exp_001").mkdir(parents=True)

        first = log.create_log_directory("student")
        second = log.create_log_directory("student")
        teacher = log.create_log_directory("teacher")

        self.assertEqual(first, Path("runs") / "student" / "exp_002")
        self.assertEqual(second, Path("runs") / "student" / "exp_003")
        self.assertEqual(teacher, Path("runs") / "teacher" / "exp_001")
        for folder in (first, second, teacher):
            self.assertTrue((self.directory / folder).is_dir())

    def test_logger_writes_everything_to_train_log_and_filters_the_console(self):
        import io
        import logging
        from utils import log

        console = io.StringIO()
        with patch("sys.stdout", console):
            logger = log.create_logger("run-folder-test", self.directory, console_level="warning")
            # Creating it again, as a resumed run does, replaces the handlers instead of doubling them
            logger = log.create_logger("run-folder-test", self.directory, console_level="warning")
        self.addCleanup(lambda: [handler.close() for handler in logger.handlers])

        logger.debug("batch details")
        logger.warning("plot not saved")
        for handler in logger.handlers:
            handler.flush()

        self.assertEqual(len(logger.handlers), 2)
        self.assertFalse(logger.propagate)
        saved = (self.directory / "train.log").read_text()
        self.assertIn("DEBUG batch details", saved)
        self.assertIn("WARNING plot not saved", saved)
        self.assertNotIn("batch details", console.getvalue())
        self.assertIn("plot not saved", console.getvalue())
        self.assertEqual(logger.handlers[1].level, logging.WARNING)


class LoadWeightsTests(unittest.TestCase):
    def test_loads_training_checkpoints_and_bare_state_dicts(self):
        from utils.model import load_weights

        trained = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            for name, content in (("wrapped.pt", {"model_state_dict": trained.state_dict(), "epoch": 4}),
                                  ("bare.pt", trained.state_dict())):
                with self.subTest(checkpoint=name):
                    path = Path(directory) / name
                    torch.save(content, path)
                    model = load_weights(torch.nn.Linear(3, 2), path, "cpu")
                    self.assertTrue(torch.equal(model.weight, trained.weight))

    def test_rejects_a_checkpoint_for_another_architecture(self):
        from utils.model import load_weights

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.pt"
            torch.save({"model_state_dict": torch.nn.Linear(4, 2).state_dict()}, path)
            with self.assertRaises(RuntimeError):
                load_weights(torch.nn.Linear(3, 2), path, "cpu")
