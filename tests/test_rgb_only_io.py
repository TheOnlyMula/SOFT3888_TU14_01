from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset_loader import DSECFusionDataset


class RgbOnlyIoTests(unittest.TestCase):
    @patch("dataset_loader.DSECDet")
    def test_rgb_only_never_reads_events_and_preserves_targets(self, dataset_class):
        base = dataset_class.return_value
        base.__getitem__.side_effect = AssertionError("Must not load the paired sample")
        base.get_events.side_effect = AssertionError("Must not read events")
        base.get_image.return_value = np.full((16, 16, 3), 255, dtype=np.uint8)
        base.get_tracks.return_value = [
            {"class_id": 2, "x": 2.0, "y": 4.0, "w": 4.0, "h": 8.0}]
        dataset = DSECFusionDataset(Path("fake"), split="test", rgb_only=True)
        images, targets = dataset[7]
        base.get_image.assert_called_once_with(7)
        base.get_tracks.assert_called_once_with(7)
        base.__getitem__.assert_not_called()
        base.get_events.assert_not_called()
        self.assertEqual(tuple(images.shape), (3, 16, 16))
        self.assertTrue(torch.equal(images, torch.ones_like(images)))
        torch.testing.assert_close(targets, torch.tensor([[2.0, 0.25, 0.5, 0.25, 0.5]]))

    @patch("dataset_loader.DSECDet")
    def test_rgb_only_accepts_empty_detection_targets(self, dataset_class):
        base = dataset_class.return_value
        base.get_image.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
        base.get_tracks.return_value = []
        dataset = DSECFusionDataset(Path("fake"), rgb_only=True)
        _, targets = dataset[0]
        self.assertEqual(targets.numel(), 0)
        base.__getitem__.assert_not_called()

    @patch("dataset_loader.DSECDet")
    def test_fusion_mode_keeps_paired_sample_and_histogram_path(self, dataset_class):
        base = dataset_class.return_value
        events = {"x": np.array([3]), "y": np.array([4]), "p": np.array([1])}
        base.__getitem__.return_value = {
            "image": np.zeros((16, 16, 3), dtype=np.uint8),
            "tracks": [], "events": events,
        }
        dataset = DSECFusionDataset(Path("fake"), rgb_only=False)
        images, histogram, targets = dataset[7]
        base.__getitem__.assert_called_once_with(7)
        base.get_image.assert_not_called()
        base.get_tracks.assert_not_called()
        self.assertEqual(tuple(images.shape), (3, 16, 16))
        self.assertEqual(tuple(histogram.shape), (2, 16, 16))
        self.assertEqual(histogram[1, 4, 3].item(), 1.0)
        self.assertEqual(histogram.sum().item(), 1.0)
        self.assertEqual(targets.numel(), 0)


if __name__ == "__main__":
    unittest.main()
