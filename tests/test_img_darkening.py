from pathlib import Path
import unittest

import numpy as np

from degradation.darkening import STRONG_ILLUMINATION, VARIATION, darken
from degradation.strength import STRENGTH_LEVELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sample_rgb_image(height=48, width=64):
    y, x = np.mgrid[:height, :width]
    image = np.stack(
        (
            80 + x % 120,
            60 + y % 140,
            100 + (x + y) % 100,
        ),
        axis=-1,
    )
    return image.astype(np.uint8)


class DarkeningTests(unittest.TestCase):
    def setUp(self):
        self.image = sample_rgb_image()

    def test_benchmark_levels_are_accepted(self):
        for level in STRENGTH_LEVELS:
            with self.subTest(level=level):
                result = darken(self.image, level, seed=17)
                self.assertEqual(result["image"].shape, self.image.shape)

    def test_strong_draws_its_illumination_around_the_strong_level(self):
        for seed in range(20):
            factor = darken(self.image, "strong", seed=seed)["global_illum_factor"]
            self.assertGreaterEqual(factor, STRONG_ILLUMINATION - VARIATION)
            self.assertLess(factor, STRONG_ILLUMINATION + VARIATION)

    def test_invalid_strength_raises_value_error(self):
        with self.assertRaises(ValueError):
            darken(self.image, "invalid_strength", seed=17)
\

    def test_negative_strength_raises_value_error(self):
        with self.assertRaises(ValueError):
            darken(self.image, -0.1, seed=17)

    def test_strength_greater_than_one_raises_value_error(self):
        with self.assertRaises(ValueError):
            darken(self.image, 1.1, seed=17)

    def test_darkened_output_is_rgb_uint8_in_valid_range(self):
        result = darken(self.image, "strong", seed=17)
        darkened = result["image"]

        self.assertEqual(darkened.shape, self.image.shape)
        self.assertEqual(darkened.dtype, np.uint8)
        self.assertGreaterEqual(int(darkened.min()), 0)
        self.assertLessEqual(int(darkened.max()), 255)

    def test_zero_strength_returns_an_unchanged_copy(self):
        result = darken(self.image, 0, seed=17)

        np.testing.assert_array_equal(result["image"], self.image)
        self.assertIsNot(result["image"], self.image)
        self.assertEqual(result["global_illum_factor"], 1.0)

    def test_same_image_strength_and_seed_are_reproducible(self):
        first = darken(self.image, "mild", seed=1234)
        second = darken(self.image, "mild", seed=1234)

        np.testing.assert_array_equal(first["image"], second["image"])
        self.assertEqual(first["global_illum_factor"], second["global_illum_factor"])

    def test_different_seeds_change_the_sensor_noise_and_result(self):
        first = darken(self.image, "mild", seed=11)
        second = darken(self.image, "mild", seed=12)

        self.assertNotEqual(first["global_illum_factor"], second["global_illum_factor"])
        self.assertFalse(np.array_equal(first["image"], second["image"]))

    def test_pre_noise_brightness_decreases_from_clean_to_mild_to_strong(self):
        clean_brightness = float(self.image.mean())
        mild = darken(self.image, "mild", seed=29)
        strong = darken(self.image, "strong", seed=29)

        self.assertGreater(clean_brightness, float(mild["mean_brightness"]))
        self.assertGreater(float(mild["mean_brightness"]), float(strong["mean_brightness"]))

    def test_post_noise_brightness_decreases_from_clean_to_mild_to_strong(self):
        clean_brightness = float(self.image.mean())
        mild = darken(self.image, "mild", seed=29)
        strong = darken(self.image, "strong", seed=29)

        self.assertGreater(clean_brightness, float(mild["image"].mean()))
        self.assertGreater(float(mild["image"].mean()), float(strong["image"].mean()))

    def test_continuous_strength_is_accepted_and_deterministic(self):
        first = darken(self.image, 0.45, seed=31)
        second = darken(self.image, 0.45, seed=31)

        self.assertEqual(first["image"].shape, self.image.shape)
        self.assertEqual(first["image"].dtype, np.uint8)
        np.testing.assert_array_equal(first["image"], second["image"])

    def test_rgb_values_are_clipped_to_0_255(self):
        result = darken(self.image, "strong", seed=17)
        darkened = result["image"]

        self.assertTrue(np.all(darkened >= 0))
        self.assertTrue(np.all(darkened <= 255))

    def test_input_image_is_not_modified(self):
        original = self.image.copy()

        darken(self.image, "strong", seed=17)

        np.testing.assert_array_equal(self.image, original)

    def test_day_and_night_example_comparisons_are_saved(self):
        examples = PROJECT_ROOT / "docs" / "darkening"
        self.assertTrue((examples / "day_000003.png").is_file())
        self.assertTrue((examples / "night_000005.png").is_file())


if __name__ == "__main__":
    unittest.main()
