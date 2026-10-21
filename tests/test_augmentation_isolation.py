from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from loaders.dataset_loader import (
    DSECFusionDataset, create_rgb_dataloaders, rgb_loader_fingerprint,
)
from evaluation.metrics import evaluate_degraded_benchmark
import training


class AugmentationIsolationTests(unittest.TestCase):
    def setUp(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        torch_state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, torch_state)
        np.random.seed(42)
        dataset_patch = patch('loaders.dataset_loader.DSECDet')
        self.base = dataset_patch.start().return_value
        self.addCleanup(dataset_patch.stop)
        self.source = (np.arange(192).reshape(8, 8, 3) % 100 + 50).astype(np.uint8)
        self.base.__len__.return_value = 4
        self.base.get_image.return_value = self.source
        self.base.get_tracks.return_value = [dict(class_id=0, x=1, y=2, w=3, h=4)]
        self.base.rel_index.side_effect = lambda idx: (
            idx, None, SimpleNamespace(root=Path('fake-sequence')))

    def test_augmentation_is_rejected_on_validation_and_test_datasets(self):
        for split in ('val', 'test'):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'training data'):
                DSECFusionDataset('fake', split=split, rgb_only=True,
                                  augmentation_shares=(0, 0, 1))

    def test_training_entry_point_keeps_validation_and_final_test_pixels_clean(self):
        runs = []

        def train(train_loader, val_loader, *args, **kwargs):
            snapshots = {'train': [], 'val': []}
            for _ in range(2):
                snapshots['train'].append(list(train_loader))
                snapshots['val'].append(list(val_loader))
            runs.append(snapshots)
            return Path('unused/best.pt')

        def evaluate(*, test_loader, **kwargs):
            runs[-1]['test'] = list(test_loader)

        with (
            patch('training.run.log'),
            patch('training.run.select_device', return_value=torch.device('cpu')),
            patch('training.run.extract_trainable_model'),
            patch('training.run.create_optimiser'),
            patch('training.run.train_model', side_effect=train),
            patch('training.run.evaluate_test_model', side_effect=evaluate),
        ):
            for shares in (None, ('0', '1', '0'), ('0', '0', '1')):
                argv = ['train', '--dataset-path', 'fake', '--batch-size', '2', '--epochs', '2']
                if shares is not None:
                    argv += ['--degradation-augmentation', '--degradation-shares', *shares]
                with patch('sys.argv', argv):
                    training.main('student', rgb_only=True)

        expected = torch.from_numpy(self.source[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
        for run in runs:
            for batches in (*run['val'], run['test']):
                for images, _ in batches:
                    self.assertTrue(torch.equal(images, expected.expand_as(images)))
        for run in runs[1:]:
            for split in ('val', 'test'):
                baseline = runs[0][split][0] if split == 'val' else runs[0][split]
                augmented = run[split][0] if split == 'val' else run[split]
                for (_, expected_targets), (_, actual_targets) in zip(baseline, augmented):
                    self.assertTrue(torch.equal(expected_targets, actual_targets))
            self.assertFalse(torch.equal(run['train'][0][0][0], runs[0]['train'][0][0][0]))

    def test_explicit_validation_degradation_is_independent_of_training_augmentation(self):
        _, clean_val = create_rgb_dataloaders('fake', MagicMock())
        clean = rgb_loader_fingerprint(clean_val)
        for condition in ('dark', 'overexposed'):
            with self.subTest(condition=condition):
                settings = dict(degradation=condition, degradation_strength=0.35, evaluation_seed=17)
                _, plain_val = create_rgb_dataloaders('fake', MagicMock(), **settings)
                train, augmented_val = create_rgb_dataloaders(
                    'fake', MagicMock(), augmentation_shares=(0, 1, 0),
                    dark_strength_range=(0.8, 1), **settings)
                list(train)
                self.assertEqual(rgb_loader_fingerprint(plain_val), rgb_loader_fingerprint(augmented_val))
                self.assertNotEqual(clean, rgb_loader_fingerprint(augmented_val))

    def test_explicit_benchmark_conditions_still_work_after_augmented_training(self):
        settings = dict(
            dataset_path=Path('fake'), model=MagicMock(), device=torch.device('cpu'),
            logger=MagicMock(), checkpoint_path=Path('unused/best.pt'), rgb_only=True,
            batch_size=2, num_workers=0, degradation_strength=0.35, evaluation_seed=17,
        )
        def evaluate(loader, *args, **kwargs):
            return dict(map50=0.5, map50_95=0.3, recall=0.6, fingerprint=rgb_loader_fingerprint(loader))

        with patch('evaluation.metrics.evaluate_test_model', side_effect=evaluate):
            before = evaluate_degraded_benchmark(**settings)
            train, _ = create_rgb_dataloaders(
                'fake', MagicMock(), augmentation_shares=(0, 1, 0), dark_strength_range=(0.8, 1))
            list(train)
            after = evaluate_degraded_benchmark(**settings)
        self.assertEqual(before, after)
        self.assertEqual(len({result['fingerprint'] for result in after.values()}), 3)
