import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler

from dataset_loader import yolo_rgb_collate_fn
from train_rgb_student import create_optimiser, create_rgb_dataloader, extract_trainable_model, fetch_best_dict, load_checkpoint, select_device


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


    ## Checkpoint loading tests
    # Test that the checkpoint returns the correct epoch and validation loss
    def test_checkpoint_returns_epoch_validation_loss(self):

        # Create original model and optimiser
        original_model = torch.nn.Linear(10, 2)
        original_optimiser = torch.optim.AdamW(original_model.parameters(), lr=0.001)

        # Do one optimiser step so AdamW has some state
        x = torch.randn(4, 10)
        y = torch.randn(4, 2)

        loss = torch.nn.functional.mse_loss(original_model(x), y)

        original_optimiser.zero_grad()
        loss.backward()
        original_optimiser.step()

        checkpoint = {'epoch': 5,
                      'model_state_dict': original_model.state_dict(),
                      'optimiser_state_dict': original_optimiser.state_dict(),
                      'val_loss': 0.42}

        # Save fake checkpoint to a temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "latest_student_yolo.pt"
            torch.save(checkpoint, checkpoint_path)

            # Create new model and optimiser to load the checkpoint into
            new_model = torch.nn.Linear(10, 2)
            new_optimiser = torch.optim.AdamW(new_model.parameters(), lr=0.001)

            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")


            # Test epoch & validation loss
            self.assertEqual(loaded_checkpoint["epoch"], 5)
            self.assertEqual(loaded_checkpoint["val_loss"], 0.42)

    # Test that the checkpoint loads the model weights & optimiser state correctly
    def test_checkpoint_loads_model_weights(self):
        # Create original model and optimiser
        original_model = torch.nn.Linear(10, 2)
        original_optimiser = torch.optim.AdamW(original_model.parameters(), lr=0.001)

        # Do one optimiser step so AdamW has some state
        x = torch.randn(4, 10)
        y = torch.randn(4, 2)

        loss = torch.nn.functional.mse_loss(original_model(x), y)

        original_optimiser.zero_grad()
        loss.backward()
        original_optimiser.step()

        checkpoint = {'epoch': 5,
                      'model_state_dict': original_model.state_dict(),
                      'optimiser_state_dict': original_optimiser.state_dict(),
                      'val_loss': 0.42}

        # Save fake checkpoint to a temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "latest_student_yolo.pt"
            torch.save(checkpoint, checkpoint_path)

            # Create new model and optimiser to load the checkpoint into
            new_model = torch.nn.Linear(10, 2)
            new_optimiser = torch.optim.AdamW(new_model.parameters(), lr=0.001)

            # Load checkpoint (model & optimiser are updated in-place)
            loaded_checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"), new_model, new_optimiser)

            # Test that the model weights are the same
            for orig_param, new_param in zip(original_model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(orig_param.data, new_param.data))

            # Test that the optimiser state is the same
            for orig_state, new_state in zip(original_optimiser.state_dict()['param_groups'], new_optimiser.state_dict()['param_groups']):
                self.assertEqual(orig_state, new_state)


    def test_missing_checkpoint_returns_default_values(self):
        model = torch.nn.Linear(10, 2)
        optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)

        result = load_checkpoint("this_file_does_not_exist.pt", torch.device("cpu"), model, optimiser)
        self.assertEqual(result.get('past_epochs'), 0)
        self.assertEqual(result.get('latest_val_loss'), float('inf'))

    ## Best model tests
    # Test saving
    def test_best_model_saving(self):
        # Create a simple model and optimiser
        model = torch.nn.Linear(10, 2)
        optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)

        # Save the best model to a temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            best_model_path = Path(temp_dir) / "best_student_yolo.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'epoch': 3,
                'val_loss': 0.25
            }, best_model_path)

            # Load the best model
            loaded_checkpoint = torch.load(best_model_path, map_location="cpu")

            # Verify that the loaded checkpoint has the expected keys and values
            self.assertIn('model_state_dict', loaded_checkpoint)
            self.assertIn('optimiser_state_dict', loaded_checkpoint)
            self.assertIn('epoch', loaded_checkpoint)
            self.assertIn('val_loss', loaded_checkpoint)

            self.assertEqual(loaded_checkpoint['epoch'], 3)
            self.assertEqual(loaded_checkpoint['val_loss'], 0.25)

    # Test loading
    def test_best_model_loading(self):
        # Create a simple model and optimiser
        model = torch.nn.Linear(10, 2)
        optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)

        # Save the best model to a temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            best_model_path = Path(temp_dir) / "best_student_yolo.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'epoch': 3,
                'val_loss': 0.25
            }, best_model_path)

            # Create new model and optimiser to load the best checkpoint into
            new_model = torch.nn.Linear(10, 2)
            new_optimiser = torch.optim.AdamW(new_model.parameters(), lr=0.001)

            # Load the best checkpoint
            loaded_checkpoint = fetch_best_dict(best_model_path, torch.device("cpu"), new_model, new_optimiser, mode=1)

            # Verify that the loaded checkpoint has the expected keys and values
            self.assertIn('best_epoch', loaded_checkpoint)
            self.assertIn('best_val_loss', loaded_checkpoint)

            self.assertEqual(loaded_checkpoint['best_epoch'], 3)
            self.assertEqual(loaded_checkpoint['best_val_loss'], 0.25)

if __name__ == "__main__":
    unittest.main()
