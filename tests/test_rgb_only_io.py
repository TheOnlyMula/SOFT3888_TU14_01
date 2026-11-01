import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from loaders import dataset_loader
from loaders.dataset_loader import (
    EVENT_SCALE_K, DSECFusionDataset, load_causal_events, read_event_window, snapshot_directory_caches,
)
from dsec_det.io import extract_from_h5_by_timewindow


class RgbOnlyIoTests(unittest.TestCase):
    @patch("loaders.dataset_loader.DSECDet")
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

    @patch("loaders.dataset_loader.DSECDet")
    def test_images_are_converted_from_bgr_to_rgb(self, dataset_class):
        base = dataset_class.return_value
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[..., 0], bgr[..., 1], bgr[..., 2] = 10, 20, 30  # cv2 order: blue, green, red
        base.get_image.return_value = bgr
        base.get_tracks.return_value = []
        images, _ = DSECFusionDataset(Path("fake"), rgb_only=True)[0]
        self.assertEqual((images[:, 0, 0] * 255).round().tolist(), [30.0, 20.0, 10.0])

    @patch("loaders.dataset_loader.DSECDet")
    def test_rgb_only_accepts_empty_detection_targets(self, dataset_class):
        base = dataset_class.return_value
        base.get_image.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
        base.get_tracks.return_value = []
        dataset = DSECFusionDataset(Path("fake"), rgb_only=True)
        _, targets = dataset[0]
        self.assertEqual(targets.numel(), 0)
        base.__getitem__.assert_not_called()

    @patch("loaders.dataset_loader.event_time_offset_us", return_value=0)
    @patch("loaders.dataset_loader.read_event_window")
    @patch("loaders.dataset_loader.DSECDet")
    def test_fusion_mode_uses_only_events_before_the_image(self, dataset_class, extract, offset):
        base = dataset_class.return_value
        base.get_image.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
        base.get_tracks.return_value = []
        directory = base.rel_index.return_value[2]
        directory.images.timestamps = np.array([1_000_000, 1_050_000, 1_100_000])
        base.rel_index.return_value = (2, None, directory)
        extract.return_value = {"x": np.array([3]), "y": np.array([4]), "p": np.array([1])}

        dataset = DSECFusionDataset(Path("fake"), rgb_only=False)
        fused, targets = dataset[7]

        base.rel_index.assert_called_once_with(7)
        # Window ends at the image timestamp and starts 50 ms earlier: no future events.
        extract.assert_called_once_with(directory.events.event_file, 1_050_000, 1_100_000)
        base.__getitem__.assert_not_called()
        base.get_image.assert_called_once_with(7)
        base.get_tracks.assert_called_once_with(7)
        histogram = fused[3:]
        self.assertEqual(tuple(fused[:3].shape), (3, 16, 16))
        self.assertEqual(tuple(histogram.shape), (2, 16, 16))
        self.assertAlmostEqual(
            histogram[1, 4, 3].item(), math.log1p(1.0) / math.log1p(EVENT_SCALE_K), places=6
        )
        self.assertEqual(int((histogram > 0).sum()), 1)
        self.assertEqual(targets.numel(), 0)

    @patch("loaders.dataset_loader.DSECDet")
    def test_event_channels_are_scaled_into_the_unit_range(self, dataset_class):
        dataset = DSECFusionDataset(Path("fake"), rgb_only=False)
        events = {
            "x": np.array([0, 1, 1] + [2] * 500),
            "y": np.array([0, 0, 0] + [0] * 500),
            "p": np.array([1, 1, 1] + [1] * 500),
        }

        histogram = dataset.generate_event_histogram(events, height=4, width=4)

        # One event is well above zero, and a pixel far past K is clipped at 1.0.
        self.assertAlmostEqual(histogram[1, 0, 0].item(), math.log1p(1.0) / math.log1p(EVENT_SCALE_K), places=6)
        self.assertAlmostEqual(histogram[1, 0, 1].item(), math.log1p(2.0) / math.log1p(EVENT_SCALE_K), places=6)
        self.assertEqual(histogram[1, 0, 2].item(), 1.0)
        self.assertGreaterEqual(histogram.min().item(), 0.0)
        self.assertLessEqual(histogram.max().item(), 1.0)

    @patch("loaders.dataset_loader.DSECDet")
    def test_histogram_counts_match_per_pixel_accumulation(self, dataset_class):
        # DSEC dtypes (uint16 coordinates, uint8 polarity) and a non-square frame, so an index
        # that overflowed or swapped height and width would land events on the wrong pixel
        height, width, n = 6, 9, 5000
        rng = np.random.default_rng(0)
        events = {
            "x": rng.integers(0, width, n).astype(np.uint16),
            "y": rng.integers(0, height, n).astype(np.uint16),
            "p": rng.integers(0, 2, n).astype(np.uint8),
        }
        expected = np.zeros((2, height, width), dtype=np.float32)
        np.add.at(expected, (events["p"].astype(int), events["y"].astype(int), events["x"].astype(int)), 1)
        expected = torch.log1p(torch.from_numpy(expected)) / math.log1p(EVENT_SCALE_K)

        histogram = DSECFusionDataset(Path("fake"), rgb_only=False).generate_event_histogram(
            events, height=height, width=width
        )

        torch.testing.assert_close(histogram, expected.clamp(max=1.0), rtol=0, atol=0)

    @patch("loaders.dataset_loader.event_time_offset_us", return_value=1_020_000)
    @patch("loaders.dataset_loader.read_event_window")
    @patch("loaders.dataset_loader.DSECDet")
    def test_event_window_is_clipped_to_recording_start(self, dataset_class, extract, offset):
        base = dataset_class.return_value
        directory = base.rel_index.return_value[2]
        directory.images.timestamps = np.array([1_040_000])
        base.rel_index.return_value = (0, None, directory)
        extract.return_value = {"x": np.array([], dtype=int)}

        load_causal_events(base, 0)

        extract.assert_called_once_with(directory.events.event_file, 1_020_000, 1_040_000)


def write_event_file(path, t_offset=5_000_000, duration_ms=40, n=3000, seed=0):
    """A small DSEC-style event file: sorted relative timestamps plus a millisecond index."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, duration_ms * 1000, n)).astype(np.uint32)
    with h5py.File(path, "w") as h5f:
        h5f["events/t"] = t
        h5f["events/x"] = rng.integers(0, 640, n).astype(np.uint16)
        h5f["events/y"] = rng.integers(0, 480, n).astype(np.uint16)
        h5f["events/p"] = rng.integers(0, 2, n).astype(np.uint8)
        h5f["ms_to_idx"] = np.searchsorted(t, np.arange(duration_ms) * 1000).astype(np.uint64)
        h5f["t_offset"] = t_offset


class EventFileReadingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(self.close_event_files)  # runs first: Windows cannot delete an open file
        self.path = Path(directory.name) / "events.h5"
        write_event_file(self.path)

    @staticmethod
    def close_event_files():
        for handles in dataset_loader._open_event_files.values():
            for h5f in handles.values():
                h5f.close()
        dataset_loader._open_event_files.clear()

    def test_selects_the_same_events_as_dsec_det(self):
        offset = 5_000_000
        windows = [
            (offset, offset + 10_000),               # recording start
            (offset + 12_345, offset + 31_999),      # windows not on millisecond boundaries
            (offset + 20_000, offset + 70_000),      # end past the last millisecond index
        ]
        for t_min, t_max in windows:
            with self.subTest(window=(t_min, t_max)):
                expected = extract_from_h5_by_timewindow(self.path, t_min, t_max)
                events = read_event_window(self.path, t_min, t_max)
                self.assertEqual(set(events), {"x", "y", "p"})
                for key in ("x", "y", "p"):
                    np.testing.assert_array_equal(events[key], expected[key])

    def test_reuses_one_handle_per_process(self):
        with patch("loaders.dataset_loader.h5py.File", wraps=h5py.File) as open_file:
            read_event_window(self.path, 5_000_000, 5_010_000)
            read_event_window(self.path, 5_010_000, 5_020_000)
            self.assertEqual(open_file.call_count, 1)
            # A forked worker has a new process ID and opens its own handle
            with patch("loaders.dataset_loader.os.getpid", return_value=-1):
                read_event_window(self.path, 5_000_000, 5_010_000)
            self.assertEqual(open_file.call_count, 2)


class DirectorySnapshotTests(unittest.TestCase):
    def test_snapshot_keeps_each_sequence_values_without_reloading(self):
        loads = []

        class Images:
            root = Path("seq/images")
            timestamps = property(lambda self: loads.append("timestamps") or np.array([1, 2]))
            image_files_distorted = property(lambda self: loads.append("files") or [Path("0.png")])

        directory = SimpleNamespace(
            images=Images(),
            tracks=SimpleNamespace(root=Path("seq/tracks"), tracks=np.array([7])),
            events=SimpleNamespace(root=Path("seq/events"), event_file=Path("seq/events.h5")),
        )
        snapshot_directory_caches(SimpleNamespace(directories={"seq": directory}))
        loads.clear()

        for _ in range(3):
            np.testing.assert_array_equal(directory.images.timestamps, [1, 2])
            self.assertEqual(directory.images.image_files_distorted, [Path("0.png")])
        self.assertEqual(loads, [])
        self.assertEqual(directory.tracks.tracks.tolist(), [7])
        self.assertEqual(directory.events.event_file, Path("seq/events.h5"))
        self.assertEqual(directory.images.root, Path("seq/images"))


if __name__ == "__main__":
    unittest.main()
