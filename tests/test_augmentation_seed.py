"""Dedicated augmentation streams and saved run configuration (criterion 4)."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from dataset_loader import (
    DSECFusionDataset, DEFAULT_DEGRADATION_SHARES, validate_augmentation_seed,
    yolo_collate_fn,
)
import training


class FakeRGBFrames:
    """Picklable fixture so tests also exercise spawn/forkserver workers."""
    def __len__(self):
        return 24

    def get_image(self, idx):
        return np.full((4, 4, 3), 100, dtype=np.uint8)

    def get_tracks(self, idx):
        return []

    def rel_index(self, idx):
        return idx, None, SimpleNamespace(root=Path('sequence'))


def augmented_dataset(seed=42, evaluation_seed=0, **kwargs):
    with patch('dataset_loader.DSECDet', return_value=FakeRGBFrames()):
        return DSECFusionDataset(
            'fake', rgb_only=True, augmentation_shares=DEFAULT_DEGRADATION_SHARES,
            augmentation_seed=seed, evaluation_seed=evaluation_seed, **kwargs)


class AugmentationSeedTests(unittest.TestCase):
    def setUp(self):
        numpy_state = np.random.get_state()
        self.addCleanup(np.random.set_state, numpy_state)
        torch_state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, torch_state)

    def test_same_seed_repeats_images_despite_evaluation_seed_and_global_rng_use(self):
        first = augmented_dataset(42, evaluation_seed=3)
        second = augmented_dataset(42, evaluation_seed=999)
        different = augmented_dataset(43)
        images = []
        repeated = []
        changed = []
        for _ in range(64):
            images.append(first[0][0])
            np.random.seed(123)
            np.random.random(100)
            torch.rand(100)
            repeated.append(second[0][0])
            changed.append(different[0][0])
        self.assertTrue(torch.equal(torch.stack(images), torch.stack(repeated)))
        self.assertFalse(torch.equal(torch.stack(images), torch.stack(changed)))
        self.assertGreater(len({image.numpy().tobytes() for image in images}), 10)

    def test_augmentation_does_not_advance_global_rngs(self):
        dataset = augmented_dataset()
        np.random.seed(123)
        torch.manual_seed(123)
        expected_numpy = np.random.random(10)
        expected_torch = torch.rand(10)
        np.random.seed(123)
        torch.manual_seed(123)
        for _ in range(10):
            dataset[0]
        np.testing.assert_array_equal(np.random.random(10), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(10), expected_torch))

    def test_evaluation_images_ignore_augmentation_seed(self):
        def evaluation_image(augmentation_seed, evaluation_seed):
            with patch('dataset_loader.DSECDet', return_value=FakeRGBFrames()):
                dataset = DSECFusionDataset(
                    'fake', split='test', rgb_only=True, degradation='dark',
                    augmentation_seed=augmentation_seed, evaluation_seed=evaluation_seed)
            return dataset[0][0]
        self.assertTrue(torch.equal(evaluation_image(1, 5), evaluation_image(99, 5)))
        self.assertFalse(torch.equal(evaluation_image(1, 5), evaluation_image(1, 6)))

    def test_persistent_workers_reproduce_and_advance_independent_streams(self):
        def run(seed, workers):
            dataset = augmented_dataset(seed)
            # Parent access must not cause all workers to inherit one RNG stream.
            dataset[0]
            loader = DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=workers,
                persistent_workers=workers > 0, collate_fn=yolo_collate_fn)
            epochs = [torch.cat([images for images, _ in loader]) for _ in range(2)]
            del loader
            return epochs
        for workers in (0, 2):
            with self.subTest(workers=workers):
                first = run(42, workers)
                torch.manual_seed(567)
                np.random.seed(567)
                repeated = run(42, workers)
                changed = run(43, workers)
                for expected, actual in zip(first, repeated):
                    self.assertTrue(torch.equal(expected, actual))
                self.assertFalse(torch.equal(first[0], first[1]))
                self.assertFalse(torch.equal(first[0], changed[0]))
                if workers:
                    self.assertFalse(torch.equal(first[0][::2], first[0][1::2]))

    def test_seed_validation_and_cli(self):
        for seed in (-1, 1.5, True, '42'):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                validate_augmentation_seed(seed)
        with patch('sys.argv', ['train', '--dataset-path', 'fake']):
            self.assertEqual(training.parse_args('test').augmentation_seed, 0)
        with patch('sys.argv', ['train', '--dataset-path', 'fake', '--augmentation-seed', '123']):
            self.assertEqual(training.parse_args('test').augmentation_seed, 123)
        with patch('sys.argv', ['train', '--dataset-path', 'fake', '--augmentation-seed', '-1']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            training.parse_args('test')

    def test_run_config_saves_effective_augmentation_settings(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                argv = ['train', '--dataset-path', 'fake', '--skip-test']
                expected = dict(
                    enabled=False, shares=dict(clean=0.5, dark=0.25, overexposed=0.25),
                    dark_strength_range=[0.35, 1.0], overexposed_strength_range=[0.35, 1.0], seed=0)
                if enabled:
                    argv += ['--degradation-augmentation', '--degradation-shares', '0.2', '0.3', '0.5',
                             '--dark-strength-range', '0.2', '0.4',
                             '--overexposed-strength-range', '0.6', '0.9', '--augmentation-seed', '123']
                    expected = dict(
                        enabled=True, shares=dict(clean=0.2, dark=0.3, overexposed=0.5),
                        dark_strength_range=[0.2, 0.4], overexposed_strength_range=[0.6, 0.9], seed=123)
                with (
                    patch('sys.argv', argv),
                    patch('dataset_loader.DSECDet', return_value=FakeRGBFrames()),
                    patch('training.log.create_log_directory', return_value=Path(directory)),
                    patch('training.log.create_logger', return_value=MagicMock()),
                    patch('training.select_device', return_value='cpu'),
                    patch('training.extract_trainable_model'),
                    patch('training.create_optimiser'),
                    patch('training.train_model') as train,
                ):
                    training.main('student', rgb_only=True)
                saved = yaml.safe_load((Path(directory) / 'config.yaml').read_text())
                self.assertEqual(saved['Training Augmentation'], expected)
                train_dataset = train.call_args.args[0].dataset
                self.assertEqual(train_dataset.augmentation_seed, expected['seed'])
                if enabled:
                    self.assertEqual(list(train_dataset.dark_strength_range), expected['dark_strength_range'])
                    self.assertEqual(list(train_dataset.overexposed_strength_range), expected['overexposed_strength_range'])
                    self.assertEqual(train_dataset.augmentation_shares, (0.2, 0.3, 0.5))
