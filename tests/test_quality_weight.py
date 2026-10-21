"""The deterministic event-quality weight w = clip(s_e * c_t, 0, 1) (proposal eq. 3)."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import training
import training.losses
from loaders.dataset_loader import DSECFusionDataset, event_coverage, yolo_collate_fn
from training.distillation import EVENT_DENSITY_REFERENCE, event_density, quality_weight, teacher_confidence
from evaluation.metrics import _prepare_yolo_batch
from utils.model import build_model, extract_trainable_model


def window(coverage, frame=0):
    return {"sequence": "zurich_city_00_a", "frame": frame, "coverage": torch.tensor(coverage)}


def quality_batch(coverages):
    """A collated batch of 64 x 64 RGB + event samples, alternately with and without an object."""
    torch.manual_seed(0)
    samples = [
        (
            torch.rand(5, 64, 64),
            torch.tensor([[2.0, 0.4, 0.5, 0.3, 0.4]]) if i % 2 == 0 else torch.zeros(0, 5),
            window(coverage, frame=i),
        )
        for i, coverage in enumerate(coverages)
    ]
    return yolo_collate_fn(samples)


def models():
    student = extract_trainable_model(ch=3, weights_path=None, device="cpu")
    teacher = build_model(5).eval().requires_grad_(False)
    teacher.end2end = True  # as load_teacher sets it
    return student, teacher


class QualityWeightTests(unittest.TestCase):
    def test_event_density_normalises_coverage_by_the_reference_and_clips_at_one(self):
        coverage = EVENT_DENSITY_REFERENCE * torch.tensor([0.0, 0.5, 1.0, 1.5])
        torch.testing.assert_close(event_density(coverage), torch.tensor([0.0, 0.5, 1.0, 1.0]))

    def test_weight_is_the_clipped_product(self):
        s_e = torch.tensor([1.0, 0.5, 0.0, 1.0])
        c_t = torch.tensor([0.9, 0.8, 0.7, 0.0])
        torch.testing.assert_close(quality_weight(s_e, c_t), torch.tensor([0.9, 0.4, 0.0, 0.0]))

    def test_teacher_confidence_rejects_output_that_is_not_end_to_end(self):
        with self.assertRaisesRegex(ValueError, "end-to-end"):
            teacher_confidence(torch.rand(2, 12, 84))  # raw B x (4 + classes) x anchors


class EventWindowTests(unittest.TestCase):
    def test_event_coverage_counts_each_pixel_with_any_event_once(self):
        histogram = torch.zeros(2, 4, 5)
        histogram[0, 0, 0] = 1.0
        histogram[1, 0, 0] = 0.3  # both polarities: still one pixel
        histogram[1, 3, 4] = 0.1
        self.assertAlmostEqual(event_coverage(histogram).item(), 2 / 20)

    @patch("loaders.dataset_loader.load_causal_events")
    @patch("loaders.dataset_loader.DSECDet")
    def test_coverage_is_measured_before_augmentation(self, base_class, load_events):
        base = base_class.return_value
        base.get_image.return_value = np.full((8, 8, 3), 100, dtype=np.uint8)
        base.get_tracks.return_value = []
        base.rel_index.return_value = (7, None, SimpleNamespace(root=Path("zurich_city_00_a")))
        load_events.return_value = {
            "x": np.array([0, 1, 2, 3]), "y": np.zeros(4, dtype=np.uint16), "p": np.zeros(4, dtype=np.uint8),
        }
        dataset = DSECFusionDataset("fake", rgb_only=False, augment=True, with_event_window=True)

        def drop_all_events(image, events, boxes, classes):
            return image, np.zeros_like(events), boxes, classes

        with patch("loaders.dataset_loader.augment_sample", side_effect=drop_all_events):
            fused, _, event_window = dataset[0]

        self.assertEqual(fused[3:].count_nonzero(), 0)  # augmentation reached the teacher input
        self.assertAlmostEqual(event_window["coverage"].item(), 4 / 64)  # but not the coverage
        self.assertEqual((event_window["sequence"], event_window["frame"]), ("zurich_city_00_a", 7))

    @patch("loaders.dataset_loader.DSECDet")
    def test_event_windows_need_the_event_channels(self, base_class):
        with self.assertRaises(ValueError):
            DSECFusionDataset("fake", rgb_only=True, with_event_window=True)

    def test_collate_keeps_plain_batches_and_gathers_event_windows(self):
        plain = yolo_collate_fn([(torch.rand(3, 8, 8), torch.zeros(0, 5))] * 2)
        self.assertEqual(len(plain), 2)

        imgs, targets, event_windows = yolo_collate_fn(
            [(torch.rand(5, 8, 8), torch.zeros(0, 5), window(c, frame=i)) for i, c in enumerate((0.1, 0.2))]
        )
        self.assertEqual(imgs.shape, (2, 5, 8, 8))
        self.assertEqual(event_windows["sequence"], ["zurich_city_00_a"] * 2)
        torch.testing.assert_close(event_windows["frame"], torch.tensor([0, 1]))
        torch.testing.assert_close(event_windows["coverage"], torch.tensor([0.1, 0.2]))


class QualityWeightedLossTests(unittest.TestCase):
    def losses(self, mode, coverages):
        """The batch distillation loss, quality and per-image object loss, for the same seeded models."""
        torch.manual_seed(0)
        student, teacher = models()
        imgs, targets, event_windows = quality_batch(coverages)
        batch = _prepare_yolo_batch(imgs, targets, torch.device("cpu"))
        per_image = []
        original = training.losses.object_distillation_loss

        def record(*args):
            per_image.append(original(*args))
            return per_image[-1]

        with patch.object(training.losses, "object_distillation_loss", side_effect=record):
            _, _, distill_loss, quality = training.losses.batch_losses(
                student, batch, teacher, mode, event_coverage=event_windows["coverage"]
            )
        return distill_loss, quality, per_image[0]

    def test_batch_loss_is_the_mean_of_each_images_weighted_object_loss(self):
        coverages = [0.0, 0.5 * EVENT_DENSITY_REFERENCE, EVENT_DENSITY_REFERENCE, 1.0]
        distill_loss, quality, per_image_loss = self.losses("object-quality", coverages)

        torch.testing.assert_close(quality["s_e"], torch.tensor([0.0, 0.5, 1.0, 1.0]))
        torch.testing.assert_close(quality["w"], (quality["s_e"] * quality["c_t"]).clamp(0, 1))
        torch.testing.assert_close(distill_loss, (quality["w"] * per_image_loss).mean())
        self.assertFalse(quality["w"].requires_grad)

    def test_object_mode_is_object_quality_with_every_weight_one(self):
        with patch.object(training.losses, "teacher_confidence", return_value=torch.ones(4)):
            weighted, quality, _ = self.losses("object-quality", [1.0] * 4)
        unweighted, no_quality, _ = self.losses("object", [1.0] * 4)

        torch.testing.assert_close(quality["w"], torch.ones(4))
        torch.testing.assert_close(weighted, unweighted)
        self.assertIsNone(no_quality)


class QualityWeightTrainingTests(unittest.TestCase):
    def test_each_epoch_records_every_images_s_e_c_t_and_w(self):
        from test_post_training_evaluation import val_results

        student, teacher = models()
        samples = [
            (torch.rand(5, 64, 64), torch.tensor([[2.0, 0.4, 0.5, 0.3, 0.4]]), window(0.5 * EVENT_DENSITY_REFERENCE, i))
            for i in range(4)
        ]
        loader = DataLoader(samples, batch_size=2, collate_fn=yolo_collate_fn)
        with tempfile.TemporaryDirectory() as directory, \
                patch("training.loop.validate_model", return_value=val_results(1.0, map50_95=0.1)):
            training.train_model(
                loader, loader, student, torch.optim.SGD(student.parameters(), lr=0.01), torch.device("cpu"),
                MagicMock(), total_epochs=2, checkpoint_path=Path(directory) / "best.pt",
                latest_path=Path(directory) / "latest.pt", teacher=teacher, lambda_distill=1.0,
                distill_mode="object-quality", nominal_batch_size=2, warmup_epochs=0,
            )
            with open(Path(directory) / "quality_weights.csv", newline="") as file:
                rows = list(csv.DictReader(file))
            with open(Path(directory) / "results.csv", newline="") as file:
                results = list(csv.DictReader(file))

        self.assertEqual(len(rows), 8)  # 4 images x 2 epochs
        self.assertEqual(
            list(rows[0]), ["epoch", "sequence", "frame", "event_density", "teacher_confidence", "quality_weight"]
        )
        self.assertEqual(sorted(int(row["frame"]) for row in rows if row["epoch"] == "1"), [0, 1, 2, 3])
        for row in rows:
            self.assertAlmostEqual(float(row["event_density"]), 0.5, places=5)
            self.assertAlmostEqual(
                float(row["quality_weight"]), 0.5 * float(row["teacher_confidence"]), places=5
            )
        epoch_weights = [float(row["quality_weight"]) for row in rows if row["epoch"] == "2"]
        self.assertAlmostEqual(float(results[1]["train/quality_weight_mean"]), np.mean(epoch_weights), places=5)

    def test_object_quality_run_reads_event_windows_and_records_the_reference(self):
        loader = MagicMock()
        create_loaders = MagicMock(return_value=(loader, loader))
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            training.run,
            select_device=MagicMock(return_value=torch.device("cpu")),
            create_dataloaders=create_loaders,
            extract_trainable_model=MagicMock(),
            load_teacher=MagicMock(),
            create_optimiser=MagicMock(),
            train_model=MagicMock(return_value=Path("best.pt")),
        ), patch.object(training.run.log, "create_log_directory", return_value=Path(directory)), \
                patch.object(training.run.log, "create_logger", return_value=MagicMock()), \
                patch.object(training.run.log, "log_command"), \
                patch.object(training.run.log, "log_model_config") as log_config, \
                patch("sys.argv", ["train", "--dataset-path", "fake", "--skip-test",
                                   "--teacher-checkpoint", "t.pt", "--distill-mode", "object-quality"]):
            training.main(role="distill", rgb_only=True)

        rgb_call, teacher_call = create_loaders.call_args_list
        self.assertTrue(teacher_call.kwargs["with_event_window"])
        self.assertNotIn("with_event_window", rgb_call.kwargs)  # validation stays RGB-only
        self.assertEqual(log_config.call_args.args[0]["Event Density Reference"], EVENT_DENSITY_REFERENCE)


if __name__ == "__main__":
    unittest.main()
