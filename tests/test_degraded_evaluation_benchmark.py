import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import random
import numpy as np
import torch
from evaluate import evaluate_degraded_benchmark, format_benchmark_table
from torch.utils.data import DataLoader, TensorDataset

from dataset_loader import DSECFusionDataset, degradation_seed, rgb_loader_fingerprint


class DegradedEvaluationBenchmarkTests(unittest.TestCase):
    def test_degradation_seed_is_deterministic(self):
        first = degradation_seed(
            evaluation_seed=42,
            sequence_name="zurich_city_00_a",
            frame_index=123,
            degradation="dark",
            strength=0.5,
        )

        second = degradation_seed(
            evaluation_seed=42,
            sequence_name="zurich_city_00_a",
            frame_index=123,
            degradation="dark",
            strength=0.5,
        )

        self.assertEqual(first, second)

    def test_degradation_seed_changes_with_frame(self):
        first = degradation_seed(
            42,
            "zurich_city_00_a",
            123,
            "dark",
            0.5,
        )

        second = degradation_seed(
            42,
            "zurich_city_00_a",
            124,
            "dark",
            0.5,
        )

        self.assertNotEqual(first, second)

    def test_degradation_seed_changes_with_evaluation_seed(self):
        first = degradation_seed(
            42,
            "zurich_city_00_a",
            123,
            "dark",
            0.5,
        )

        second = degradation_seed(
            43,
            "zurich_city_00_a",
            123,
            "dark",
            0.5,
        )

        self.assertNotEqual(first, second)

    def test_degradation_seed_changes_with_sequence(self):
        first = degradation_seed(
            42,
            "zurich_city_00_a",
            123,
            "dark",
            0.5,
        )

        second = degradation_seed(
            42,
            "zurich_city_01_a",
            123,
            "dark",
            0.5,
        )

        self.assertNotEqual(first, second)

    def test_degradation_seed_changes_with_settings(self):
        first = degradation_seed(
            42,
            "zurich_city_00_a",
            123,
            "dark",
            0.5,
        )

        different_type = degradation_seed(
            42,
            "zurich_city_00_a",
            123,
            "overexposed",
            0.5,
        )

        different_strength = degradation_seed(
            42,
            "zurich_city_00_a",
            123,
            "dark",
            0.8,
        )

        self.assertNotEqual(first, different_type)
        self.assertNotEqual(first, different_strength)

    @patch("dataset_loader.DSECDet")
    @patch("dataset_loader.yaml_file_to_dict")
    def test_invalid_degradation_is_rejected(self, _, __):
        with self.assertRaises(ValueError):
            DSECFusionDataset(
                Path("fake"),
                degradation="rain",
            )

    @patch("dataset_loader.DSECDet")
    @patch("dataset_loader.yaml_file_to_dict")
    def test_negative_degradation_strength_is_rejected(self, _, __):
        with self.assertRaises(ValueError):
            DSECFusionDataset(
                Path("fake"),
                degradation="dark",
                degradation_strength=-0.1,
            )

    def test_rgb_fingerprint_is_independent_of_batch_size(self):
        images = torch.arange(
            6 * 3 * 4 * 4,
            dtype=torch.float32,
        ).reshape(6, 3, 4, 4)

        targets = torch.zeros(6)

        dataset = TensorDataset(images, targets)

        loader_small = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        loader_large = DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
        )

        self.assertEqual(
            rgb_loader_fingerprint(loader_small),
            rgb_loader_fingerprint(loader_large),
        )

    def test_rgb_fingerprint_is_independent_of_prior_random_use(self):
        images = torch.arange(
            6 * 3 * 4 * 4,
            dtype=torch.float32,
        ).reshape(6, 3, 4, 4)

        targets = torch.zeros(6)

        dataset = TensorDataset(images, targets)

        first_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
        )

        first = rgb_loader_fingerprint(first_loader)

        random.random()
        np.random.random(100)
        torch.rand(100)

        second_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
        )

        second = rgb_loader_fingerprint(second_loader)

        self.assertEqual(first, second)

    @patch("evaluate.evaluate_test_model")
    @patch("evaluate.rgb_loader_fingerprint")
    @patch("evaluate.create_test_loader")
    def test_benchmark_runs_all_three_conditions(
        self,
        create_loader,
        fingerprint,
        evaluate_model,
    ):
        clean_loader = object()
        dark_loader = object()
        over_loader = object()

        create_loader.side_effect = [
            clean_loader,
            dark_loader,
            over_loader,
        ]

        fingerprint.side_effect = [
            "clean-fingerprint",
            "dark-fingerprint",
            "over-fingerprint",
        ]

        evaluate_model.side_effect = [
            {
                "map50": 0.60,
                "map50_95": 0.40,
                "recall": 0.70,
            },
            {
                "map50": 0.50,
                "map50_95": 0.30,
                "recall": 0.60,
            },
            {
                "map50": 0.45,
                "map50_95": 0.25,
                "recall": 0.55,
            },
        ]

        logger = MagicMock()
        model = MagicMock()
        device = torch.device("cpu")

        results = evaluate_degraded_benchmark(
            dataset_path=Path("fake"),
            model=model,
            device=device,
            logger=logger,
            checkpoint_path=Path("best.pt"),
            rgb_only=True,
            batch_size=4,
            num_workers=0,
            degradation_strength=0.5,
            evaluation_seed=42,
        )

        self.assertEqual(
            [call.kwargs["degradation"] for call in create_loader.call_args_list],
            ["clean", "dark", "overexposed"],
        )

        self.assertEqual(create_loader.call_count, 3)

        for call in create_loader.call_args_list:
            self.assertTrue(call.kwargs["rgb_only"])
            self.assertEqual(call.kwargs["degradation_strength"], 0.5)
            self.assertEqual(call.kwargs["evaluation_seed"], 42)

        self.assertAlmostEqual(results["dark"]["map50_drop"], 0.10)
        self.assertAlmostEqual(results["overexposed"]["map50_drop"], 0.15)
        self.assertAlmostEqual(results["dark"]["map50_95_drop"], 0.10)
        self.assertAlmostEqual(results["overexposed"]["map50_95_drop"], 0.15)
        self.assertEqual(results["clean"]["map50_drop"], 0.0)
        self.assertEqual(results["clean"]["map50_95_drop"], 0.0)

    @patch("evaluate.evaluate_test_model")
    @patch("evaluate.rgb_loader_fingerprint")
    @patch("evaluate.create_test_loader")
    def test_benchmark_uses_same_checkpoint_for_every_condition(
        self,
        create_loader,
        fingerprint,
        evaluate_model,
    ):
        create_loader.side_effect = [object(), object(), object()]
        fingerprint.side_effect = ["clean", "dark", "overexposed"]

        evaluate_model.return_value = {
            "map50": 0.5,
            "map50_95": 0.3,
            "recall": 0.6,
        }

        checkpoint = Path("best_student.pt")

        evaluate_degraded_benchmark(
            dataset_path=Path("fake"),
            model=MagicMock(),
            device=torch.device("cpu"),
            logger=MagicMock(),
            checkpoint_path=checkpoint,
            rgb_only=True,
            evaluation_seed=42,
        )

        self.assertEqual(evaluate_model.call_count, 3)

        for call in evaluate_model.call_args_list:
            self.assertEqual(call.args[4], checkpoint)

    def test_benchmark_table_contains_required_metrics(self):
        results = {
            "clean": {
                "map50": 0.60,
                "map50_95": 0.40,
                "recall": 0.70,
                "map50_drop": 0.0,
                "map50_95_drop": 0.0,
            },
            "dark": {
                "map50": 0.50,
                "map50_95": 0.30,
                "recall": 0.60,
                "map50_drop": 0.10,
                "map50_95_drop": 0.10,
            },
            "overexposed": {
                "map50": 0.45,
                "map50_95": 0.25,
                "recall": 0.55,
                "map50_drop": 0.15,
                "map50_95_drop": 0.15,
            },
        }

        lines = format_benchmark_table(results)
        output = "\n".join(lines)

        self.assertIn("Condition", output)
        self.assertIn("mAP50", output)
        self.assertIn("mAP50-95", output)
        self.assertIn("Recall", output)
        self.assertIn("mAP50 Drop", output)
        self.assertIn("mAP50-95 Drop", output)
        self.assertIn("clean", output)
        self.assertIn("dark", output)
        self.assertIn("overexposed", output)

    @patch("dataset_loader.DSECDet")
    def test_rgb_benchmark_does_not_open_event_files(self, dataset_class):
        base = dataset_class.return_value

        base.get_events.side_effect = AssertionError("Must not read events")
        base.__getitem__.side_effect = AssertionError("Must not load paired sample")

        base.get_image.return_value = np.full(
            (16, 16, 3),
            100,
            dtype=np.uint8,
        )
        base.get_tracks.return_value = []

        directory = base.rel_index.return_value[2]
        directory.name = "zurich_city_00_a"
        base.rel_index.return_value = (7, None, directory)

        dataset = DSECFusionDataset(
            Path("fake"),
            split="test",
            rgb_only=True,
            degradation="dark",
            degradation_strength=0.5,
            evaluation_seed=42,
        )

        dataset[7]

        base.get_image.assert_called_once_with(7)
        base.get_tracks.assert_called_once_with(7)
        base.get_events.assert_not_called()
        base.__getitem__.assert_not_called()

    @patch("dataset_loader.load_causal_events")
    @patch("dataset_loader.DSECDet")
    def test_teacher_event_channels_and_labels_are_unchanged(
        self,
        dataset_class,
        load_events,
    ):
        base = dataset_class.return_value

        base.get_image.return_value = np.full(
            (16, 16, 3),
            100,
            dtype=np.uint8,
        )

        base.get_tracks.return_value = [
            {
                "class_id": 2,
                "x": 2.0,
                "y": 4.0,
                "w": 4.0,
                "h": 8.0,
            }
        ]

        load_events.return_value = {
            "x": np.array([3]),
            "y": np.array([4]),
            "p": np.array([1]),
        }

        directory = base.rel_index.return_value[2]
        directory.name = "zurich_city_00_a"
        base.rel_index.return_value = (7, None, directory)

        clean_dataset = DSECFusionDataset(
            Path("fake"),
            split="test",
            rgb_only=False,
            degradation="clean",
        )

        degraded_dataset = DSECFusionDataset(
            Path("fake"),
            split="test",
            rgb_only=False,
            degradation="dark",
            degradation_strength=0.5,
            evaluation_seed=42,
        )

        clean_image, clean_targets = clean_dataset[7]
        degraded_image, degraded_targets = degraded_dataset[7]

        torch.testing.assert_close(
            clean_image[3:],
            degraded_image[3:],
        )

        torch.testing.assert_close(
            clean_targets,
            degraded_targets,
        )

    def test_rgb_fingerprint_is_independent_of_worker_count(self):
        images = torch.arange(
            8 * 3 * 4 * 4,
            dtype=torch.float32,
        ).reshape(8, 3, 4, 4)

        targets = torch.zeros(8)

        dataset = TensorDataset(images, targets)

        loader_zero_workers = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        loader_four_workers = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=4,
        )

        self.assertEqual(
            rgb_loader_fingerprint(loader_zero_workers),
            rgb_loader_fingerprint(loader_four_workers),
        )


if __name__ == "__main__":
    unittest.main()
