import unittest
from pathlib import Path
from unittest.mock import patch

from torch.utils.data import SequentialSampler

import overfit


class OverfitTests(unittest.TestCase):
    @patch("overfit.DSECFusionDataset", return_value=list(range(100)))
    def test_tiny_loader_uses_fixed_training_frames(self, dataset_class):
        loader = overfit.tiny_loader(Path("data"), num_samples=8, batch_size=4)

        dataset_class.assert_called_once_with(dataset_path=Path("data"), split="train", rgb_only=True)
        self.assertEqual(loader.dataset.indices, [0, 12, 25, 37, 50, 62, 75, 87])
        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertEqual(loader.batch_size, 4)


if __name__ == "__main__":
    unittest.main()