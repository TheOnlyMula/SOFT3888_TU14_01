import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from dataset_loader import limit_dataset, yolo_collate_fn
from utils import DSEC_NAMES, build_model, set_loss_schedule
from dataset_loader import create_rgb_dataloaders
from evaluate import validate_model
from ultralytics.optim import MuSGD
from utils import (
    OPTIMISER_LR,
    create_optimiser,
    extract_trainable_model,
    fetch_best_dict,
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
        with patch("utils.torch.cuda.is_available", return_value=True):
            self.assertEqual(select_device().type, "cuda")
        with patch("utils.torch.cuda.is_available", return_value=False):
            self.assertEqual(select_device().type, "cpu")

    @patch("dataset_loader.DSECFusionDataset")
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

    @patch("utils.build_model")
    def test_extract_trainable_model(self, mock_build_model):
        model = MagicMock(spec=torch.nn.Module)
        parameters = [
            torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False),
            torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False),
        ]
        model.parameters.return_value = parameters
        model.modules.return_value = []
        model.to.return_value = model
        mock_build_model.return_value = model

        result = extract_trainable_model(
            ch=3,
            weights_path="dummy.pt",
            device=torch.device("cpu"),
        )

        mock_build_model.assert_called_once_with(3, "dummy.pt")
        model.train.assert_called_once()
        model.to.assert_called_once_with(torch.device("cpu"))
        self.assertTrue(all(parameter.requires_grad for parameter in parameters))
        self.assertIsInstance(result.args, SimpleNamespace)
        self.assertEqual(result.args.box, 7.5)
        self.assertEqual(result.args.cls, 0.5)
        self.assertEqual(result.args.dfl, 1.5)
        self.assertEqual(result.args.device, "cpu")

    def test_build_model_matches_dsec_setup(self):
        for ch in (3, 5):
            with self.subTest(ch=ch):
                model = build_model(ch=ch)
                self.assertEqual(model.model[0].conv.in_channels, ch)
                self.assertEqual(model.model[-1].nc, 8)
                self.assertEqual(model.names, DSEC_NAMES)

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

    def test_checkpoint_loads_epoch_model_and_optimiser(self):
        original_model = torch.nn.Linear(10, 2)
        original_optimiser = torch.optim.AdamW(original_model.parameters(), lr=0.001)
        loss = torch.nn.functional.mse_loss(
            original_model(torch.randn(4, 10)), torch.randn(4, 2)
        )
        original_optimiser.zero_grad()
        loss.backward()
        original_optimiser.step()

        checkpoint = {
            "epoch": 5,
            "model_state_dict": original_model.state_dict(),
            "optimiser_state_dict": original_optimiser.state_dict(),
            "val_loss": 0.42,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_student_yolo.pt"
            torch.save(checkpoint, path)
            model = torch.nn.Linear(10, 2)
            optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)
            loaded = load_checkpoint(path, torch.device("cpu"), model, optimiser)

        self.assertEqual(loaded, {"past_epochs": 5, "latest_val_loss": 0.42})
        for original, restored in zip(original_model.parameters(), model.parameters()):
            self.assertTrue(torch.equal(original, restored))
        self.assertEqual(
            original_optimiser.state_dict()["param_groups"],
            optimiser.state_dict()["param_groups"],
        )

    def test_missing_checkpoint_returns_default_values(self):
        model = torch.nn.Linear(10, 2)
        optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)
        result = load_checkpoint(
            "this_file_does_not_exist.pt",
            torch.device("cpu"),
            model,
            optimiser,
        )
        self.assertEqual(result, {"past_epochs": 0, "latest_val_loss": float("inf")})

    def test_fetch_best_checkpoint_can_restore_state(self):
        original_model = torch.nn.Linear(10, 2)
        original_optimiser = torch.optim.AdamW(original_model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best_student_yolo.pt"
            torch.save(
                {
                    "model_state_dict": original_model.state_dict(),
                    "optimiser_state_dict": original_optimiser.state_dict(),
                    "epoch": 3,
                    "val_loss": 0.25,
                    "val_metrics": {"map50_95": 0.4},
                },
                path,
            )
            model = torch.nn.Linear(10, 2)
            optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)
            result = fetch_best_dict(
                path,
                torch.device("cpu"),
                model,
                optimiser,
                mode=1,
            )

        self.assertEqual(result, {"best_epoch": 3, "best_map50_95": 0.4})
        for original, restored in zip(original_model.parameters(), model.parameters()):
            self.assertTrue(torch.equal(original, restored))

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
        import log

        def row(epoch, loss):
            return {"epoch": epoch, "train/loss": loss, "val/loss": loss,
                    "metrics/mAP50": 0.1 * epoch, "metrics/mAP50-95": 0.05 * epoch, "lr": 0.01}

        with tempfile.TemporaryDirectory() as directory:
            for epoch, loss in ((1, 3.0), (2, 2.0), (3, 1.5)):
                log.log_epoch_results(row(epoch, loss), Path(directory))
            # A resumed run repeats epoch 3 after an interruption
            log.log_epoch_results(row(3, 1.0), Path(directory))

            with open(Path(directory) / "results.csv", newline="") as file:
                saved = list(csv.DictReader(file))
            self.assertEqual([r["epoch"] for r in saved], ["1", "2", "3"])
            self.assertEqual(float(saved[-1]["train/loss"]), 1.0)
            self.assertTrue((Path(directory) / "results.png").is_file())


class LoadWeightsTests(unittest.TestCase):
    def test_loads_training_checkpoints_and_bare_state_dicts(self):
        from utils import load_weights

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
        from utils import load_weights

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.pt"
            torch.save({"model_state_dict": torch.nn.Linear(4, 2).state_dict()}, path)
            with self.assertRaises(RuntimeError):
                load_weights(torch.nn.Linear(3, 2), path, "cpu")
