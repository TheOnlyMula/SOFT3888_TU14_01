import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from dataset_loader import yolo_collate_fn
from train_rgb_student import (
    create_optimiser,
    create_rgb_dataloaders,
    extract_trainable_model,
    fetch_best_dict,
    load_checkpoint,
    select_device,
    validate_model,
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
                call(dataset_path=Path("fake-dataset"), split="train", rgb_only=True),
                call(dataset_path=Path("fake-dataset"), split="val", rgb_only=True),
            ],
        )

    def test_invalid_dataloader_configuration_is_logged(self):
        logger = MagicMock()
        for kwargs in ({"batch_size": 0}, {"batch_size": 33}, {"num_workers": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                create_rgb_dataloaders(Path("fake"), logger=logger, **kwargs)
        self.assertEqual(logger.error.call_count, 3)

    @patch("utils.YOLO")
    def test_extract_trainable_model(self, mock_yolo_class):
        model = MagicMock(spec=torch.nn.Module)
        parameters = [
            torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False),
            torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False),
        ]
        model.parameters.return_value = parameters
        model.modules.return_value = []
        model.to.return_value = model
        model.args = {}
        wrapper = mock_yolo_class.return_value
        wrapper.model = model

        result = extract_trainable_model(
            yaml_path="student.yaml",
            weights_path="dummy.pt",
            device=torch.device("cpu"),
        )

        mock_yolo_class.assert_called_once_with("student.yaml", task="detect")
        wrapper.load.assert_called_once_with("dummy.pt")
        model.train.assert_called_once()
        model.to.assert_called_once_with(torch.device("cpu"))
        self.assertTrue(all(parameter.requires_grad for parameter in parameters))
        self.assertIsInstance(result.args, SimpleNamespace)
        self.assertEqual(result.args.box, 7.5)
        self.assertEqual(result.args.cls, 0.5)
        self.assertEqual(result.args.dfl, 1.5)
        self.assertEqual(result.args.device, "cpu")

    def test_optimiser_configuration(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        model[0].requires_grad_(False)
        optimiser = create_optimiser(model)

        self.assertIsInstance(optimiser, torch.optim.AdamW)
        self.assertEqual(optimiser.param_groups[0]["lr"], 1e-4)
        self.assertEqual(optimiser.param_groups[0]["weight_decay"], 0.0005)

        optimised = {
            id(parameter)
            for group in optimiser.param_groups
            for parameter in group["params"]
        }
        unfrozen = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        frozen = {id(parameter) for parameter in model.parameters() if not parameter.requires_grad}
        self.assertEqual(optimised, unfrozen)
        self.assertTrue(optimised.isdisjoint(frozen))

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

        self.assertEqual(result, {"best_epoch": 3, "best_val_loss": 0.25})
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

        def model_call(batch):
            gradient_states.append(torch.is_grad_enabled())
            return torch.tensor([1.5]), {
                "box_loss": torch.tensor(0.5),
                "cls_loss": torch.tensor(0.4),
                "l1_loss": torch.tensor(0.3),
            }

        model.side_effect = model_call
        losses = validate_model(
            [(images, targets)],
            model,
            torch.device("cpu"),
            logger,
        )

        model.eval.assert_called_once()
        self.assertEqual(gradient_states, [False])
        self.assertEqual(losses, (1.5, 0.5, 0.4000000059604645, 0.30000001192092896))


if __name__ == "__main__":
    unittest.main()
