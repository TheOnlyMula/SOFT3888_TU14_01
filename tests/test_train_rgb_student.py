import unittest
from pathlib import Path
from unittest.mock import patch

from torch.utils.data import RandomSampler

from dataset_loader import yolo_rgb_collate_fn
from train_rgb_student import create_rgb_dataloader, select_device


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


if __name__ == "__main__":
    unittest.main()
