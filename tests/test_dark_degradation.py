"""The benchmark's darkened condition and the training augmentation both use darkening.darken."""
import unittest
from unittest.mock import patch

import numpy as np

import darkening
from dataset_loader import DARKEST_ILLUMINATION, apply_dark_degradation, augment_rgb_degradation

DARK_ONLY = (0.0, 1.0, 0.0)


def street_image():
    rng = np.random.default_rng(0)
    return rng.integers(40, 220, size=(48, 64, 3), dtype=np.uint8)


class DarkDegradationTests(unittest.TestCase):
    def test_strength_maps_linearly_onto_darkening_illumination(self):
        image = street_image()
        with patch("dataset_loader.darken", wraps=darkening.darken) as darken:
            for strength, illumination in ((1.0, DARKEST_ILLUMINATION), (0.35, 0.72), (0.5, 0.6)):
                apply_dark_degradation(image, strength=strength, seed=3)
                self.assertAlmostEqual(darken.call_args.args[1], illumination)
        self.assertEqual(DARKEST_ILLUMINATION, darkening.STRENGTH_RANGES["strong"])

    def test_zero_strength_returns_an_unchanged_copy_without_darkening(self):
        image = street_image()
        with patch("dataset_loader.darken") as darken:
            result = apply_dark_degradation(image, strength=0.0, seed=3)
        darken.assert_not_called()
        np.testing.assert_array_equal(result, image)
        self.assertIsNot(result, image)

    def test_same_seed_repeats_and_another_seed_differs(self):
        image = street_image()
        first = apply_dark_degradation(image, strength=1.0, seed=11)
        np.testing.assert_array_equal(first, apply_dark_degradation(image, strength=1.0, seed=11))
        self.assertFalse(np.array_equal(first, apply_dark_degradation(image, strength=1.0, seed=12)))

    def test_output_is_a_darker_uint8_image_of_the_same_shape(self):
        image = street_image()
        mild = apply_dark_degradation(image, strength=0.35, seed=5)
        strong = apply_dark_degradation(image, strength=1.0, seed=5)
        for result in (mild, strong):
            self.assertEqual(result.dtype, np.uint8)
            self.assertEqual(result.shape, image.shape)
        self.assertLess(strong.mean(), mild.mean())
        self.assertLess(mild.mean(), image.mean())

    def test_training_darkening_is_reproducible_from_the_augmentation_stream(self):
        image = street_image()
        first = augment_rgb_degradation(image, DARK_ONLY, rng=np.random.default_rng(9))
        again = augment_rgb_degradation(image, DARK_ONLY, rng=np.random.default_rng(9))
        other = augment_rgb_degradation(image, DARK_ONLY, rng=np.random.default_rng(10))
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))


if __name__ == "__main__":
    unittest.main()
