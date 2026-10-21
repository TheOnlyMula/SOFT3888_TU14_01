import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from torch.utils.data import SequentialSampler

from training import overfit


class OverfitTests(unittest.TestCase):
    @patch("training.overfit.DSECFusionDataset", return_value=list(range(100)))
    def test_tiny_loader_uses_fixed_training_frames(self, dataset_class):
        loader = overfit.tiny_loader(Path("data"), num_samples=8, batch_size=4)

        dataset_class.assert_called_once_with(dataset_path=Path("data"), split="train", rgb_only=True)
        self.assertEqual(loader.dataset.indices, [0, 12, 25, 37, 50, 62, 75, 87])
        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertEqual(loader.batch_size, 4)

    @patch("training.overfit.train_model")
    @patch("training.overfit.create_optimiser")
    @patch("training.overfit.extract_trainable_model")
    @patch("training.overfit.log")
    @patch("training.overfit.tiny_loader")
    def test_trains_and_validates_on_the_same_loader(self, tiny_loader, *_):
        argv = ["overfit.py", "--dataset-path", "data", "--epochs", "50"]
        with patch.object(sys, "argv", argv):
            overfit.main()

        tiny_loader.assert_called_once_with(Path("data"), 8, 4)
        args, kwargs = overfit.train_model.call_args
        self.assertIs(args[0], tiny_loader.return_value)
        self.assertIs(args[1], tiny_loader.return_value)
        self.assertEqual(kwargs["total_epochs"], 50)


if __name__ == "__main__":
    unittest.main()