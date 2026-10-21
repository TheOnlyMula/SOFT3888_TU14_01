import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from loaders.dataset_loader import DSECFusionDataset, create_rgb_dataloaders
from degradation.conditions import (
    DEFAULT_DEGRADATION_SHARES, DEFAULT_STRENGTH_RANGE, augment_rgb_degradation,
    validate_degradation_shares, validate_strength_range,
)
from degradation.strength import STRENGTH_LEVELS
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

    @patch('loaders.dataset_loader.DSECDet')
    def test_loaded_images_draw_uniform_strengths_from_condition_range(self, base_class):
        base_class.return_value.__len__.return_value = 4
        base_class.return_value.get_image.return_value = np.full((2, 2, 3), 100, dtype=np.uint8)
        base_class.return_value.get_tracks.return_value = []
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        np.random.seed(73)
        for shares, transform, bounds in (
            ((0, 1, 0), 'darken', (0.2, 0.4)),
            ((0, 0, 1), 'overexpose', (0.6, 0.9)),
        ):
            with self.subTest(transform=transform):
                train, _ = create_rgb_dataloaders(
                    'fake', MagicMock(), augmentation_shares=shares,
                    dark_strength_range=(0.2, 0.4), overexposed_strength_range=(0.6, 0.9))
                with patch('degradation.conditions.' + transform,
                           side_effect=lambda image, *args, **kwargs:
                           {'image': image} if transform == 'darken' else image) as degrade:
                    for _ in range(2000):
                        train.dataset[0]
                strengths = np.array([
                    call.kwargs['strength'] if transform == 'darken' else call.args[1]
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
                'degradation.conditions.darken', return_value={'image': 'dark'}
            ), patch('degradation.conditions.overexpose', return_value='overexposed'):
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

    @patch('loaders.dataset_loader.DSECDet')
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

    @patch('loaders.dataset_loader.DSECDet')
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


class EventSampleAugmentationTests(unittest.TestCase):
    @patch('loaders.dataset_loader.load_causal_events')
    @patch('loaders.dataset_loader.DSECDet')
    def test_degradation_changes_only_the_rgb_of_event_samples(self, base_class, load_events):
        from pathlib import Path
        from types import SimpleNamespace

        base = base_class.return_value
        base.__len__.return_value = 4
        base.get_image.return_value = np.full((8, 8, 3), 100, dtype=np.uint8)
        base.get_tracks.return_value = []
        base.rel_index.return_value = (0, None, SimpleNamespace(root=Path('sequence')))
        load_events.return_value = {
            'x': np.array([1, 2]), 'y': np.array([1, 2], dtype=np.uint16), 'p': np.array([0, 1], dtype=np.uint8),
        }
        always_dark = dict(augmentation_shares=(0, 1, 0), augmentation_seed=5)

        student_rgb, _ = DSECFusionDataset('fake', rgb_only=True, **always_dark)[0]
        teacher_input, _ = DSECFusionDataset('fake', rgb_only=False, **always_dark)[0]
        clean_input, _ = DSECFusionDataset('fake', rgb_only=False)[0]

        # The teacher's RGB is darkened exactly like the RGB-only student's, and its events stay clean
        torch.testing.assert_close(teacher_input[:3], student_rgb, rtol=0, atol=0)
        self.assertLess(teacher_input[:3].mean(), clean_input[:3].mean())
        torch.testing.assert_close(teacher_input[3:], clean_input[3:], rtol=0, atol=0)


class RgbDropoutTests(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        from types import SimpleNamespace

        base_patch = patch('loaders.dataset_loader.DSECDet')
        events_patch = patch('loaders.dataset_loader.load_causal_events')
        base = base_patch.start().return_value
        events_patch.start().return_value = {
            'x': np.array([1, 2]), 'y': np.array([1, 2], dtype=np.uint16), 'p': np.array([0, 1], dtype=np.uint8),
        }
        self.addCleanup(base_patch.stop)
        self.addCleanup(events_patch.stop)
        base.__len__.return_value = 4
        base.get_image.return_value = np.full((8, 8, 3), 100, dtype=np.uint8)
        base.get_tracks.return_value = [dict(class_id=2, x=1, y=1, w=4, h=4)]
        base.rel_index.return_value = (0, None, SimpleNamespace(root=Path('sequence')))

    def test_dropout_blanks_only_the_rgb_of_about_that_share_of_samples(self):
        clean_input, clean_targets = DSECFusionDataset('fake', rgb_only=False)[0]
        dataset = DSECFusionDataset('fake', rgb_only=False, rgb_dropout=0.3, augmentation_seed=1)
        blanked = 0
        for _ in range(1000):
            sample, targets = dataset[0]
            torch.testing.assert_close(sample[3:], clean_input[3:], rtol=0, atol=0)  # events kept
            torch.testing.assert_close(targets, clean_targets)  # labels kept
            if sample[:3].count_nonzero() == 0:
                blanked += 1
            else:
                torch.testing.assert_close(sample[:3], clean_input[:3], rtol=0, atol=0)
        self.assertAlmostEqual(blanked / 1000, 0.3, delta=0.05)

    def test_dropout_needs_events_and_training_data(self):
        for kwargs in (dict(rgb_only=True), dict(rgb_only=False, split='val'), dict(rgb_only=False, split='test'),
                       dict(rgb_only=False, rgb_dropout=1.0)):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                DSECFusionDataset('fake', **{'rgb_dropout': 0.15, **kwargs})

    def test_a_student_run_refuses_rgb_dropout(self):
        import tempfile
        from pathlib import Path
        import training

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(training.run.log, 'create_log_directory', return_value=Path(directory)), \
                patch.object(training.run.log, 'create_logger', return_value=MagicMock()), \
                patch.object(training.run.log, 'log_command'), \
                patch('sys.argv', ['train_rgb_student.py', '--dataset-path', 'fake', '--rgb-dropout', '0.15']), \
                self.assertRaisesRegex(ValueError, 'only applies to the teacher'):
            training.main(role='student', rgb_only=True)
