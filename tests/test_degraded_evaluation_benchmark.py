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

        self.assertAlmostEqual(results["dark"]["map50_drop"], 0.10)
        self.assertAlmostEqual(results["overexposed"]["map50_drop"], 0.15)

    def test_benchmark_table_contains_required_metrics(self):
        results = {
            "clean": {
                "map50": 0.60,
                "map50_95": 0.40,
                "recall": 0.70,
                "map50_drop": 0.0,
            },
            "dark": {
                "map50": 0.50,
                "map50_95": 0.30,
                "recall": 0.60,
                "map50_drop": 0.10,
            },
            "overexposed": {
                "map50": 0.45,
                "map50_95": 0.25,
                "recall": 0.55,
                "map50_drop": 0.15,
            },
        }

        lines = format_benchmark_table(results)
        output = "\n".join(lines)

        self.assertIn("Condition", output)
        self.assertIn("mAP50", output)
        self.assertIn("mAP50-95", output)
        self.assertIn("Recall", output)
        self.assertIn("mAP50 Drop", output)
        self.assertIn("clean", output)
        self.assertIn("dark", output)
        self.assertIn("overexposed", output)


if __name__ == "__main__":
    unittest.main()
