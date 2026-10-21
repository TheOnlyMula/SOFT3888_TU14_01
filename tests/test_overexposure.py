"""Unit tests for degradation/overexposure.py.

Run from the repository root:
    python -m unittest discover -s tests -p "test_overexposure.py" -v
"""

import unittest

import numpy as np

from degradation.overexposure import MAX_BRIGHTNESS_OFFSET, MIN_GAMMA, overexpose
from degradation.strength import STRENGTH_LEVELS


def ramp_image() -> np.ndarray:
    """1 x 256 x 3 image containing every uint8 value in every channel."""
    values = np.arange(256, dtype=np.uint8)
    return np.repeat(values.reshape(1, 256, 1), 3, axis=2)


def random_image(seed: int = 0, shape=(48, 64, 3)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def reference_lut(strength: float) -> np.ndarray:
    """Independent re-implementation of the documented formula as a 256-entry table."""
    x = np.arange(256, dtype=np.float64) / 255.0
    gamma = 1.0 - (1.0 - MIN_GAMMA) * strength
    offset = MAX_BRIGHTNESS_OFFSET * strength
    return np.rint(np.clip(x ** gamma + offset, 0.0, 1.0) * 255.0).astype(np.uint8)


class TestPresets(unittest.TestCase):
    def test_named_levels_match_readme_table(self):
        self.assertEqual(STRENGTH_LEVELS, {"mild": 0.35, "strong": 1.0})
        self.assertEqual(MIN_GAMMA, 0.4)
        self.assertEqual(MAX_BRIGHTNESS_OFFSET, 0.4)

    def test_named_level_equals_its_numeric_value(self):
        image = random_image()
        for name, value in STRENGTH_LEVELS.items():
            with self.subTest(level=name):
                np.testing.assert_array_equal(
                    overexpose(image, name), overexpose(image, value)
                )


class TestFormula(unittest.TestCase):
    def test_matches_documented_formula_for_every_pixel_value(self):
        image = ramp_image()
        for strength in (0.1, 0.35, 0.5, 0.6, 0.9, 1.0):
            with self.subTest(strength=strength):
                expected = reference_lut(strength)
                output = overexpose(image, strength)
                for channel in range(3):
                    np.testing.assert_array_equal(output[0, :, channel], expected)

    def test_known_values_strong(self):
        output = overexpose(ramp_image(), "strong")[0, :, 0]
        self.assertEqual(output[0], 102)  # Black is lifted by the 0.4 offset.
        self.assertEqual(output[70], 254)
        # x ** 0.4 + 0.4 rounds to 255 from about x = 0.6 ** 2.5 (pixel 71).
        self.assertTrue(np.all(output[71:] == 255))

    def test_known_values_mild(self):
        output = overexpose(ramp_image(), "mild")[0, :, 0]
        self.assertEqual(output[0], 36)  # round(0.14 * 255)
        self.assertEqual(output[210], 254)
        self.assertTrue(np.all(output[211:] == 255))

    def test_white_stays_white(self):
        image = np.full((4, 4, 3), 255, dtype=np.uint8)
        for strength in (0.01, "mild", "strong"):
            with self.subTest(strength=strength):
                self.assertTrue(np.all(overexpose(image, strength) == 255))

    def test_channels_are_processed_independently(self):
        image = random_image(seed=3)
        output = overexpose(image, 0.7)
        for channel in range(3):
            with self.subTest(channel=channel):
                single = np.repeat(image[:, :, channel : channel + 1], 3, axis=2)
                np.testing.assert_array_equal(
                    overexpose(single, 0.7)[:, :, 0], output[:, :, channel]
                )

    def test_grey_pixels_stay_grey(self):
        output = overexpose(ramp_image(), 0.5)
        np.testing.assert_array_equal(output[..., 0], output[..., 1])
        np.testing.assert_array_equal(output[..., 1], output[..., 2])


class TestMonotonicity(unittest.TestCase):
    def test_mean_brightness_increases_from_clean_to_mild_to_strong(self):
        image = random_image(seed=14)
        clean_mean = float(image.mean())
        mild_mean = float(overexpose(image, "mild").mean())
        strong_mean = float(overexpose(image, "strong").mean())
        self.assertLess(clean_mean, mild_mean)
        self.assertLess(mild_mean, strong_mean)

    def test_never_darkens_a_pixel(self):
        image = random_image(seed=1)
        for strength in (0.05, 0.35, 0.8, 1.0):
            with self.subTest(strength=strength):
                self.assertTrue(np.all(overexpose(image, strength) >= image))

    def test_brighter_input_gives_brighter_or_equal_output(self):
        for strength in (0.2, 0.35, 1.0):
            with self.subTest(strength=strength):
                output = overexpose(ramp_image(), strength)[0, :, 0].astype(int)
                self.assertTrue(np.all(np.diff(output) >= 0))

    def test_stronger_setting_gives_brighter_or_equal_output(self):
        image = random_image(seed=2)
        previous = image.astype(int)
        for strength in np.linspace(0.0, 1.0, 11):
            current = overexpose(image, float(strength)).astype(int)
            self.assertTrue(np.all(current >= previous), f"strength={strength}")
            previous = current

    def test_strong_saturates_more_than_mild(self):
        image = random_image(seed=4)

        def saturated_fraction(img):
            return np.mean(np.all(img == 255, axis=2))

        mild = saturated_fraction(overexpose(image, "mild"))
        strong = saturated_fraction(overexpose(image, "strong"))
        self.assertGreater(strong, mild)
        self.assertGreater(mild, saturated_fraction(image))


class TestOutputContract(unittest.TestCase):
    def test_shape_and_dtype_preserved(self):
        for shape in ((1, 1, 3), (5, 7, 3), (480, 640, 3)):
            with self.subTest(shape=shape):
                output = overexpose(np.zeros(shape, dtype=np.uint8), 0.5)
                self.assertEqual(output.shape, shape)
                self.assertEqual(output.dtype, np.uint8)

    def test_input_is_not_modified(self):
        image = random_image(seed=5)
        original = image.copy()
        for strength in (0.0, "mild", "strong"):
            overexpose(image, strength)
        np.testing.assert_array_equal(image, original)

    def test_read_only_input_is_accepted(self):
        image = random_image(seed=6)
        image.flags.writeable = False
        overexpose(image, "strong")  # Should not raise.

    def test_output_is_a_new_array(self):
        image = random_image(seed=7)
        for strength in (0.0, 0.5):
            with self.subTest(strength=strength):
                output = overexpose(image, strength)
                self.assertIsNot(output, image)
                self.assertFalse(np.shares_memory(output, image))

    def test_zero_strength_returns_unchanged_copy(self):
        image = random_image(seed=8)
        output = overexpose(image, 0.0)
        np.testing.assert_array_equal(output, image)
        output[0, 0, 0] ^= 0xFF  # Changing the copy must not touch the input.
        self.assertNotEqual(output[0, 0, 0], image[0, 0, 0])

    def test_integer_zero_behaves_like_float_zero(self):
        image = random_image(seed=9)
        np.testing.assert_array_equal(overexpose(image, 0), image)

    def test_tiny_strength_is_nearly_identity(self):
        image = ramp_image()
        np.testing.assert_array_equal(overexpose(image, 1e-12), image)

    def test_non_contiguous_input(self):
        image = random_image(seed=10, shape=(20, 30, 3))
        view = image[::2, ::-1]  # Strided, reversed view.
        self.assertFalse(view.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(
            overexpose(view, 0.6), overexpose(np.ascontiguousarray(view), 0.6)
        )

    def test_deterministic(self):
        image = random_image(seed=11)
        first = overexpose(image, "strong")
        for _ in range(3):
            np.testing.assert_array_equal(overexpose(image, "strong"), first)

    def test_does_not_touch_global_random_state(self):
        np.random.seed(123)
        expected = np.random.rand()
        np.random.seed(123)
        overexpose(random_image(seed=12), "strong")
        self.assertEqual(np.random.rand(), expected)


class TestStrengthValidation(unittest.TestCase):
    def setUp(self):
        self.image = random_image(seed=13, shape=(4, 4, 3))

    def test_accepts_numpy_scalars(self):
        for strength in (np.float32(0.5), np.float64(0.5), np.int64(1), np.int32(0)):
            with self.subTest(strength=type(strength).__name__):
                overexpose(self.image, strength)

    def test_boundaries_are_inclusive(self):
        overexpose(self.image, 0.0)
        overexpose(self.image, 1.0)

    def test_rejects_unknown_level_names(self):
        for name in ("medium", "Strong", "STRONG", "", " mild"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    overexpose(self.image, name)

    def test_rejects_out_of_range_numbers(self):
        for strength in (-0.01, 1.01, -1, 2, 100.0):
            with self.subTest(strength=strength):
                with self.assertRaises(ValueError):
                    overexpose(self.image, strength)

    def test_rejects_non_finite_numbers(self):
        for strength in (float("nan"), float("inf"), float("-inf"), np.nan):
            with self.subTest(strength=strength):
                with self.assertRaises(ValueError):
                    overexpose(self.image, strength)

    def test_rejects_booleans(self):
        for strength in (True, False, np.bool_(True)):
            with self.subTest(strength=strength):
                with self.assertRaises(TypeError):
                    overexpose(self.image, strength)

    def test_rejects_non_numeric_types(self):
        for strength in (None, [0.5], (0.5,), {"s": 1}, 0.5j, np.array(0.5)):
            with self.subTest(strength=repr(strength)):
                with self.assertRaises(TypeError):
                    overexpose(self.image, strength)


class TestImageValidation(unittest.TestCase):
    def test_rejects_non_arrays(self):
        for image in (None, [[[0, 0, 0]]], 5, "image"):
            with self.subTest(image=repr(image)):
                with self.assertRaises(TypeError):
                    overexpose(image, 0.5)

    def test_rejects_wrong_dtype(self):
        for dtype in (np.float32, np.float64, np.uint16, np.int8, np.int32, bool):
            with self.subTest(dtype=np.dtype(dtype).name):
                with self.assertRaises(TypeError):
                    overexpose(np.zeros((4, 4, 3), dtype=dtype), 0.5)

    def test_rejects_wrong_shape(self):
        for shape in ((4, 4), (4, 4, 1), (4, 4, 4), (3, 4, 4), (1, 4, 4, 3), (12,)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    overexpose(np.zeros(shape, dtype=np.uint8), 0.5)

    def test_rejects_empty_images(self):
        for shape in ((0, 4, 3), (4, 0, 3), (0, 0, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    overexpose(np.zeros(shape, dtype=np.uint8), 0.5)

    def test_image_validated_before_strength(self):
        # A bad image should be reported even when the strength is also bad.
        with self.assertRaises(TypeError):
            overexpose(np.zeros((4, 4, 3), dtype=np.float32), "nonsense")


if __name__ == "__main__":
    unittest.main()
