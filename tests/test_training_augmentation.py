import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from dataset_loader import (
    DEFAULT_DEGRADATION_SHARES, DSECFusionDataset,
    augment_rgb_degradation, create_rgb_dataloaders, validate_degradation_shares,
)
from training import parse_args


class TrainingAugmentationTests(unittest.TestCase):
    def test_condition_shares_over_many_draws(self):
        image = np.full((1, 1, 3), 100, dtype=np.uint8)
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        np.random.seed(42)
        for shares in (DEFAULT_DEGRADATION_SHARES, (0.1, 0.3, 0.6)):
            with self.subTest(shares=shares), patch(
                'dataset_loader.apply_dark_degradation', return_value='dark'
            ), patch('dataset_loader.overexpose', return_value='overexposed'):
                counts = dict.fromkeys(('clean', 'dark', 'overexposed'), 0)
                for _ in range(20000):
                    result = augment_rgb_degradation(image, shares)
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
        with patch('dataset_loader.np.random.choice', side_effect=['clean', 'dark', 'overexposed']) as choice:
            clean, targets = dataset[0]
            dark, dark_targets = dataset[0]
            bright, bright_targets = dataset[0]
        self.assertEqual(choice.call_count, 3)
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
