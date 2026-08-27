import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler

from dataset_loader import yolo_rgb_collate_fn
from train_rgb_student import create_optimiser, create_rgb_dataloader, extract_trainable_model, select_device


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
        dataset_class.return_value = [None] * 8
        dataset_path = Path("fake-dataset")

        loader = create_rgb_dataloader(
            dataset_path=dataset_path,
            batch_size=4,
            num_workers=4,
        )

        dataset_class.assert_called_once_with(
            dataset_path=dataset_path,
            split="train",
            rgb_only=True,
        )
        self.assertEqual(loader.batch_size, 4)
        self.assertIsInstance(loader.sampler, RandomSampler)
        self.assertEqual(loader.num_workers, 4)
        self.assertTrue(loader.pin_memory)
        self.assertTrue(loader.persistent_workers)
        self.assertIs(loader.collate_fn, yolo_rgb_collate_fn)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            create_rgb_dataloader(Path("fake"), batch_size=0)

        with self.assertRaises(ValueError):
            create_rgb_dataloader(Path("fake"), batch_size=33)

        with self.assertRaises(ValueError):
            create_rgb_dataloader(Path("fake"), num_workers=-1)

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

if __name__ == "__main__":
    unittest.main()
