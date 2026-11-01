"""Acceptance check of the over-exposure calibration on the documented DSEC reference frames.

Set DSEC_ROOT to the dataset folder containing ``train/thun_00_a`` and ``train/zurich_city_09_a``.
Without that dataset, this integration test is explicitly skipped.
"""

import hashlib
import json
import os
import unittest
from pathlib import Path

import numpy as np

from degradation.overexposure import overexpose


REFERENCE_PATH = Path(__file__).resolve().parents[1] / "docs/overexposure/reference_metrics.json"
# Strong over-exposure must whiten part of a daytime frame, but not most of it: the previous
# calibration whitened about two thirds, which left nothing for any model to detect.
DAY_STRONG_SATURATION = (0.10, 0.50)


def saturated(image):
    return float(np.all(image == 255, axis=2).mean())


class TestReferenceFrames(unittest.TestCase):
    def test_strength_orders_the_reference_frames_and_strong_stays_partial(self):
        dataset_root = os.environ.get("DSEC_ROOT")
        if not dataset_root:
            self.skipTest("Set DSEC_ROOT to run the DSEC reference-frame acceptance check")

        import cv2

        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(reference["samples"]), 5)

        for sample in reference["samples"]:
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
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mild, strong = overexpose(rgb, "mild"), overexpose(rgb, "strong")

                self.assertLess(float(rgb.mean()), float(mild.mean()))
                self.assertLess(float(mild.mean()), float(strong.mean()))
                self.assertLessEqual(saturated(rgb), saturated(mild))
                self.assertLess(saturated(mild), saturated(strong))
                if sample["image_path"].startswith("train/thun_00_a/"):
                    low, high = DAY_STRONG_SATURATION
                    self.assertGreaterEqual(saturated(strong), low)
                    self.assertLessEqual(saturated(strong), high)


if __name__ == "__main__":
    unittest.main()
