"""Acceptance check for Issue #41 using the documented DSEC night frames.

Set DSEC_ROOT to the dataset folder containing ``train/zurich_city_09_a``.
Without that dataset, this integration test is explicitly skipped.
"""

import hashlib
import json
import os
import unittest
from pathlib import Path

import numpy as np

from overexposure import overexpose


REFERENCE_PATH = Path(__file__).resolve().parents[1] / "docs/overexposure/reference_metrics.json"


class TestNightReference(unittest.TestCase):
    def test_strong_meets_documented_night_saturation_floor(self):
        dataset_root = os.environ.get("DSEC_ROOT")
        if not dataset_root:
            self.skipTest("Set DSEC_ROOT to run the DSEC night-frame acceptance check")

        import cv2

        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        night_samples = [
            sample for sample in reference["samples"]
            if sample["image_path"].startswith("train/zurich_city_09_a/")
        ]
        self.assertEqual(len(night_samples), 4)
        floor = reference["proposed_night_reference_floor_percent"] / 100.0

        for sample in night_samples:
            source = Path(dataset_root) / sample["image_path"]
            with self.subTest(frame=sample["image_path"]):
                self.assertTrue(source.is_file(), f"Missing DSEC frame: {source}")
                source_bytes = source.read_bytes()
                self.assertEqual(
                    hashlib.sha256(source_bytes).hexdigest(),
                    sample["image_sha256"],
                    f"DSEC frame differs from the documented reference: {source}",
                )
                bgr = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertIsNotNone(bgr, f"Cannot decode DSEC frame: {source}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mild = overexpose(rgb, "mild")
                strong = overexpose(rgb, "strong")
                self.assertLess(float(rgb.mean()), float(mild.mean()))
                self.assertLess(float(mild.mean()), float(strong.mean()))
                clean_saturation = float(np.all(rgb == 255, axis=2).mean())
                mild_saturation = float(np.all(mild == 255, axis=2).mean())
                saturated_fraction = float(np.all(strong == 255, axis=2).mean())
                self.assertLess(clean_saturation, mild_saturation)
                self.assertLess(mild_saturation, saturated_fraction)
                self.assertGreaterEqual(saturated_fraction, floor)


if __name__ == "__main__":
    unittest.main()
