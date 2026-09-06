import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tools.rgb_image_degradation import RGBImageDegradation


class RGBImageDegradationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.dataset_path = Path(self.temp_dir.name) / "dataset"
        self.output_path = Path(self.temp_dir.name) / "output"

    def create_sequence(self, split="train", name="sequence_01", image_count=2):
        rgb_path = self.dataset_path / split / name / "images" / "left" / "distorted"
        rgb_path.mkdir(parents=True, exist_ok=True)

        for index in range(image_count):
            image = np.full((8, 8, 3), 100 + index, dtype=np.uint8)
            cv2.imwrite(str(rgb_path / f"{index:06d}.png"), image)

        return rgb_path

    def test_invalid_split_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "split must be either 'train' or 'test'",
        ):
            RGBImageDegradation(self.dataset_path, split="validation")

    def test_missing_dataset_split_is_rejected(self):
        pipeline = RGBImageDegradation(self.dataset_path, split="train")

        with self.assertRaises(FileNotFoundError):
            pipeline.validate_dataset()

    def test_get_sequences_returns_only_valid_rgb_sequences(self):
        self.create_sequence(name="valid_sequence")

        invalid_sequence = self.dataset_path / "train" / "invalid_sequence"
        invalid_sequence.mkdir(parents=True)

        pipeline = RGBImageDegradation(self.dataset_path, split="train")
        sequences = pipeline.get_sequences()

        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0]["name"], "valid_sequence")
        self.assertTrue(sequences[0]["rgb_path"].exists())

    def test_get_image_paths_returns_png_files_in_sorted_order(self):
        rgb_path = self.create_sequence(image_count=0)

        for filename in ("000002.png", "000000.png", "000001.png"):
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            cv2.imwrite(str(rgb_path / filename), image)

        pipeline = RGBImageDegradation(self.dataset_path, split="train")
        images = pipeline.get_image_paths()

        self.assertEqual(
            [sample["path"].name for sample in images],
            ["000000.png", "000001.png", "000002.png"],
        )

    def test_load_image_rejects_unreadable_file(self):
        self.create_sequence(image_count=0)
        invalid_path = self.dataset_path / "missing.png"

        pipeline = RGBImageDegradation(self.dataset_path, split="train")

        with self.assertRaisesRegex(RuntimeError, "Failed to load image"):
            pipeline.load_image(invalid_path)

    @patch("tools.rgb_image_degradation.random.uniform", return_value=0.5)
    @patch("tools.rgb_image_degradation.random.choice", return_value=True)
    def test_degrade_brightness_can_darken_image(
        self,
        mock_choice,
        mock_uniform,
    ):
        pipeline = RGBImageDegradation(self.dataset_path)
        image = np.full((8, 8, 3), 100, dtype=np.uint8)

        degraded = pipeline.degrade_brightness(image)

        self.assertTrue(np.all(degraded == 50))
        mock_uniform.assert_called_once_with(0.3, 0.7)

    @patch("tools.rgb_image_degradation.random.uniform", return_value=1.5)
    @patch("tools.rgb_image_degradation.random.choice", return_value=False)
    def test_degrade_brightness_can_brighten_image(
        self,
        mock_choice,
        mock_uniform,
    ):
        pipeline = RGBImageDegradation(self.dataset_path)
        image = np.full((8, 8, 3), 100, dtype=np.uint8)

        degraded = pipeline.degrade_brightness(image)

        self.assertTrue(np.all(degraded == 150))
        mock_uniform.assert_called_once_with(1.3, 2.0)

    @patch.object(
        RGBImageDegradation,
        "degrade_brightness",
        side_effect=lambda image: image,
    )
    def test_save_degraded_dataset_preserves_structure_and_limit(
        self,
        mock_degrade,
    ):
        self.create_sequence(name="sequence_01", image_count=3)

        pipeline = RGBImageDegradation(self.dataset_path, split="train")
        processed = pipeline.save_degraded_dataset(
            self.output_path,
            limit=2,
        )

        expected_directory = (
            self.output_path / "train" / "sequence_01" / "images" / "left" / "distorted"
        )

        self.assertEqual(processed, 2)
        self.assertTrue((expected_directory / "000000.png").exists())
        self.assertTrue((expected_directory / "000001.png").exists())
        self.assertFalse((expected_directory / "000002.png").exists())
        self.assertEqual(mock_degrade.call_count, 2)


if __name__ == "__main__":
    unittest.main()
