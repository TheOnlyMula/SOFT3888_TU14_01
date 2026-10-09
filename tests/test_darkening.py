"""darkening.darken returns the same result layout for every strength."""
import unittest

import numpy as np

from darkening import darken


class DarkenTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(0).integers(40, 220, size=(48, 64, 3), dtype=np.uint8)

    def test_zero_strength_returns_a_black_image_in_the_input_layout(self):
        result = darken(self.image, 0)
        self.assertEqual(result["image"].shape, self.image.shape)
        self.assertEqual(result["image"].dtype, np.uint8)
        self.assertFalse(result["image"].any())
        self.assertEqual(result["proportion_dark"], 1.0)

    def test_every_strength_returns_the_same_layout_and_types(self):
        reference = darken(self.image, 0.5, seed=1)
        for strength in (0, 1, "strong"):
            with self.subTest(strength=strength):
                result = darken(self.image, strength, seed=1)
                self.assertEqual(result.keys(), reference.keys())
                self.assertEqual(result["image"].shape, reference["image"].shape)
                self.assertEqual(result["image"].dtype, reference["image"].dtype)
                for key in ("global_illum_factor", "mean_brightness", "proportion_dark"):
                    self.assertIsInstance(result[key], float)


if __name__ == "__main__":
    unittest.main()
