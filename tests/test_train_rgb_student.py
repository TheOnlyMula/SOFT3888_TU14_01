import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from dataset_loader import yolo_rgb_collate_fn
from train_rgb_student import (
        create_optimiser, create_rgb_dataloaders,
        extract_trainable_model, select_device,
        validate_model,
)


class TrainRgbStudentTests(unittest.TestCase):
    def test_select_device(self):
        with patch(
            "train_rgb_student.torch.cuda.is_available",
            return_value=True,
        ):
            self.assertEqual(select_device().type, "cuda")

        with patch(
            "train_rgb_student.torch.cuda.is_available",
            return_value=False,
        ):
            self.assertEqual(select_device().type, "cpu")

    @patch("train_rgb_student.DSECFusionDataset")
    def test_dataloader_configuration(self, dataset_class):
        dataset_class.return_value = [None] * 10

        train_loader, val_loader = create_rgb_dataloaders(
            dataset_path=Path("fake-dataset"),
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

        self.assertIs(train_loader.collate_fn, yolo_rgb_collate_fn)
        self.assertIs(val_loader.collate_fn, yolo_rgb_collate_fn)

        dataset_class.assert_called_once_with(
            dataset_path=Path("fake-dataset"),
            split="train",
            rgb_only=True,
        )

    @patch("train_rgb_student.DSECFusionDataset")
    def test_dataset_split_is_reproducible(self, dataset_class):
        dataset_class.return_value = list(range(10))

        train_loader_1, val_loader_1 = create_rgb_dataloaders(
            Path("fake-dataset"),
        )

        train_loader_2, val_loader_2 = create_rgb_dataloaders(
            Path("fake-dataset"),
        )

        self.assertEqual(
            train_loader_1.dataset.indices,
            train_loader_2.dataset.indices,
        )

        self.assertEqual(
            val_loader_1.dataset.indices,
            val_loader_2.dataset.indices,
        )

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            create_rgb_dataloaders(Path("fake"), batch_size=0)

        with self.assertRaises(ValueError):
            create_rgb_dataloaders(Path("fake"), batch_size=51)

        with self.assertRaises(ValueError):
            create_rgb_dataloaders(Path("fake"), num_workers=-1)

    @patch("train_rgb_student.YOLO")
    def test_extract_trainable_model(self, mock_yolo_class):
        mock_torch_model = MagicMock(spec=torch.nn.Module)

        fake_param_1 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        fake_param_2 = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        mock_torch_model.parameters.return_value = [fake_param_1, fake_param_2]

        mock_torch_model.to.return_value = mock_torch_model

        mock_yolo_instance = mock_yolo_class.return_value
        mock_yolo_instance.model = mock_torch_model

        device = torch.device("cpu")
        result = extract_trainable_model(weights_path="dummy.pt", device=device)

        # Assert: Verify YOLO was instantiated correctly
        mock_yolo_class.assert_called_once_with("dummy.pt")

        # Assert: Verify state changes
        mock_torch_model.train.assert_called_once()
        mock_torch_model.to.assert_called_once_with(device)

        # Assert: Verify gradients were enabled
        for param in mock_torch_model.parameters():
            self.assertTrue(param.requires_grad)

        # Assert: Verify hyperparameters were patched correctly
        self.assertTrue(hasattr(result, "args"))
        self.assertIsInstance(result.args, SimpleNamespace)
        self.assertEqual(result.args.box, 7.5)
        self.assertEqual(result.args.cls, 0.5)
        self.assertEqual(result.args.dfl, 1.5)
        self.assertEqual(result.args.device, device)

    def test_optimiser_configuration(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 2),
        )

        model[0].requires_grad_(False)

        optimiser = create_optimiser(model)

        self.assertIsInstance(optimiser, torch.optim.AdamW)
        self.assertEqual(optimiser.param_groups[0]["lr"], 1e-3)
        self.assertEqual(
            optimiser.param_groups[0]["weight_decay"],
            0.0005,
        )

        optimised_parameters = {
            id(parameter)
            for group in optimiser.param_groups
            for parameter in group["params"]
        }

        unfrozen_parameters = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }

        frozen_parameters = {
            id(parameter)
            for parameter in model.parameters()
            if not parameter.requires_grad
        }

        self.assertEqual(
            optimised_parameters,
            unfrozen_parameters,
        )

        self.assertTrue(
            optimised_parameters.isdisjoint(frozen_parameters)
        )

    def test_validation_uses_eval_mode_and_disables_gradients(self):
        model = MagicMock()

        images = torch.randn(2, 3, 32, 32)
        targets = torch.tensor(
            [
                [0, 1, 0.5, 0.5, 0.2, 0.2],
                [1, 2, 0.4, 0.4, 0.1, 0.1],
            ],
            dtype=torch.float32,
        )

        val_loader = [(images, targets)]

        gradient_states = []

        def model_call(batch):
            gradient_states.append(torch.is_grad_enabled())
            return torch.tensor([1.5]), None

        model.side_effect = model_call

        val_loss = validate_model(
            val_loader,
            model,
            torch.device("cpu"),
        )

        model.eval.assert_called_once()
        self.assertEqual(gradient_states, [False])
        self.assertEqual(val_loss, 1.5)


if __name__ == "__main__":
    unittest.main()
