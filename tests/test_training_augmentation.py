import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from dataset_loader import (
    DEFAULT_DEGRADATION_SHARES, DEFAULT_STRENGTH_RANGE, DSECFusionDataset,
    augment_rgb_degradation, create_rgb_dataloaders, validate_degradation_shares,
    validate_strength_range,
)
from overexposure import STRENGTH_LEVELS
from training import parse_args


class TrainingAugmentationTests(unittest.TestCase):
    def test_strength_ranges_match_evaluation_levels(self):
        self.assertEqual(DEFAULT_STRENGTH_RANGE,
                         (STRENGTH_LEVELS['mild'], STRENGTH_LEVELS['strong']))

    def test_invalid_strength_ranges(self):
        for bounds in ((-0.1, 1), (0, 1.1), (0.8, 0.2), (0,), (0, 0.5, 1),
                       (float('nan'), 1), (0, float('inf'))):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                validate_strength_range(bounds)
        self.assertEqual(validate_strength_range((0, 0)), (0, 0))
        self.assertEqual(validate_strength_range((1, 1)), (1, 1))

    @patch('dataset_loader.DSECDet')
    def test_loaded_images_draw_uniform_strengths_from_condition_range(self, base_class):
        base_class.return_value.__len__.return_value = 4
        base_class.return_value.get_image.return_value = np.full((2, 2, 3), 100, dtype=np.uint8)
        base_class.return_value.get_tracks.return_value = []
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        np.random.seed(73)
        for shares, transform, bounds in (
            ((0, 1, 0), 'apply_dark_degradation', (0.2, 0.4)),
            ((0, 0, 1), 'overexpose', (0.6, 0.9)),
        ):
            with self.subTest(transform=transform):
                train, _ = create_rgb_dataloaders(
                    'fake', MagicMock(), augmentation_shares=shares,
                    dark_strength_range=(0.2, 0.4), overexposed_strength_range=(0.6, 0.9))
                with patch('dataset_loader.' + transform,
                           side_effect=lambda image, *args, **kwargs: image) as degrade:
                    for _ in range(2000):
                        train.dataset[0]
                strengths = np.array([
                    call.kwargs['strength'] if transform == 'apply_dark_degradation' else call.args[1]
                    for call in degrade.call_args_list
                ])
                self.assertTrue(np.all((strengths >= bounds[0]) & (strengths <= bounds[1])))
                self.assertGreater(len(np.unique(strengths)), 1900)
                counts, _ = np.histogram(strengths, bins=4, range=bounds)
                np.testing.assert_allclose(counts / len(strengths), 0.25, atol=0.04)

    def test_fixed_strength_and_clean_image(self):
        image = np.full((2, 2, 3), 100, dtype=np.uint8)
        rng = np.random.default_rng(42)
        for shares in ((0, 1, 0), (0, 0, 1)):
            result = augment_rgb_degradation(image, shares, (0, 0), (0, 0), rng=rng)
            np.testing.assert_array_equal(result, image)
        clean_rng = MagicMock(wraps=rng)
        self.assertIs(augment_rgb_degradation(image, (1, 0, 0), rng=clean_rng), image)
        clean_rng.uniform.assert_not_called()

    def test_cli_strength_ranges(self):
        with patch('sys.argv', ['train', '--dataset-path', 'fake']):
            args = parse_args('test')
        self.assertEqual(args.dark_strength_range, DEFAULT_STRENGTH_RANGE)
        self.assertEqual(args.overexposed_strength_range, DEFAULT_STRENGTH_RANGE)
        with patch('sys.argv', ['train', '--dataset-path', 'fake',
                               '--dark-strength-range', '0.2', '0.4',
                               '--overexposed-strength-range', '0.6', '0.9']):
            args = parse_args('test')
        self.assertEqual(args.dark_strength_range, (0.2, 0.4))
        self.assertEqual(args.overexposed_strength_range, (0.6, 0.9))
        for flag in ('--dark-strength-range', '--overexposed-strength-range'):
            with self.subTest(flag=flag), patch('sys.argv', [
                'train', '--dataset-path', 'fake', flag, '0.8', '0.2'
            ]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args('test')

    def test_condition_shares_over_many_draws(self):
        image = np.full((1, 1, 3), 100, dtype=np.uint8)
        rng = np.random.default_rng(42)
        for shares in (DEFAULT_DEGRADATION_SHARES, (0.1, 0.3, 0.6)):
            with self.subTest(shares=shares), patch(
                'dataset_loader.apply_dark_degradation', return_value='dark'
            ), patch('dataset_loader.overexpose', return_value='overexposed'):
                counts = dict.fromkeys(('clean', 'dark', 'overexposed'), 0)
                for _ in range(20000):
                    result = augment_rgb_degradation(image, shares, rng=rng)
                    counts['clean' if result is image else result] += 1
                for condition, expected in zip(counts, shares):
                    self.assertAlmostEqual(counts[condition] / 20000, expected, delta=0.015)

    def test_invalid_shares(self):
        for shares in ((0, 0, 0), (-0.1, 0.5, 0.6), (1, 1, 1), (1, 0),
                       (float('nan'), 0, 1), (float('inf'), 0, 0)):
            with self.subTest(shares=shares), self.assertRaises(ValueError):
                validate_degradation_shares(shares)

    @patch('dataset_loader.DSECDet')
    def test_dataset_samples_fresh_condition_and_preserves_targets(self, base_class):
        base = base_class.return_value
        base.get_image.return_value = np.full((8, 8, 3), 100, dtype=np.uint8)
        base.get_tracks.return_value = [dict(class_id=0, x=1, y=1, w=2, h=2)]
        dataset = DSECFusionDataset('fake', rgb_only=True,
                                    augmentation_shares=DEFAULT_DEGRADATION_SHARES)
        rng = MagicMock(wraps=np.random.default_rng(42))
        rng.choice.side_effect = ['clean', 'dark', 'overexposed']
        with patch.object(dataset, 'augmentation_rng', return_value=rng):
            clean, targets = dataset[0]
            dark, dark_targets = dataset[0]
            bright, bright_targets = dataset[0]
        self.assertEqual(rng.choice.call_count, 3)
        self.assertLess(dark.mean(), clean.mean())
        self.assertGreater(bright.mean(), clean.mean())
        torch.testing.assert_close(targets, dark_targets)
        torch.testing.assert_close(targets, bright_targets)
        base.get_events.assert_not_called()

    @patch('dataset_loader.DSECDet')
    def test_loader_enables_only_training_augmentation(self, base_class):
        base_class.return_value.__len__.return_value = 4
        train, val = create_rgb_dataloaders(
            'fake', MagicMock(), augmentation_shares=(0, 0, 1))
        self.assertEqual(train.dataset.augmentation_shares, (0, 0, 1))
        self.assertIsNone(val.dataset.augmentation_shares)
        plain, _ = create_rgb_dataloaders('fake', MagicMock())
        self.assertIsNone(plain.dataset.augmentation_shares)

    def test_cli_defaults_and_custom_shares(self):
        with patch('sys.argv', ['train', '--dataset-path', 'fake']):
            args = parse_args('test')
        self.assertFalse(args.degradation_augmentation)
        self.assertEqual(args.degradation_shares, DEFAULT_DEGRADATION_SHARES)
        with patch('sys.argv', ['train', '--dataset-path', 'fake',
                               '--degradation-augmentation', '--degradation-shares', '0.2', '0.3', '0.5']):
            args = parse_args('test')
        self.assertTrue(args.degradation_augmentation)
        self.assertEqual(args.degradation_shares, (0.2, 0.3, 0.5))
        with patch('sys.argv', ['train', '--dataset-path', 'fake',
                               '--degradation-shares', '1', '1', '1']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args('test')
