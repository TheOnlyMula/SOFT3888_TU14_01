"""The experiment configs in configs/ and how --config combines with command-line flags."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import yaml

import training

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
STUDENT_CONFIGS = ("rgb_baseline", "rgb_augmented", "naive_distillation", "object_distillation", "proposed_event")
DISTILLATION_CONFIGS = ("naive_distillation", "object_distillation", "proposed_event")


def parse(*argv, script="train_rgb_student.py"):
    with patch("sys.argv", [script, *argv]):
        return training.parse_args("test")


def parse_config(name, *extra):
    return parse("--config", str(CONFIGS / f"{name}.yaml"), "--dataset-path", "fake", *extra)


def settings(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def rejected(*argv):
    with contextlib.redirect_stderr(io.StringIO()), patch("sys.argv", ["train_rgb_student.py", *argv]):
        try:
            training.parse_args("test")
        except SystemExit:
            return True
    return False


class ExperimentConfigTests(unittest.TestCase):
    def test_every_config_parses_to_its_settings(self):
        for path in sorted(CONFIGS.glob("*.yaml")):
            with self.subTest(config=path.name):
                args = parse_config(path.stem)
                for name, value in settings(path.stem).items():
                    parsed = getattr(args, name)
                    if isinstance(value, list):
                        self.assertEqual(list(parsed), value)
                    elif name == "teacher_checkpoint":
                        self.assertEqual(parsed, Path(value))
                    else:
                        self.assertEqual(parsed, value)
                self.assertEqual(args.config, path)

    def test_student_comparisons_share_architecture_budget_and_augmentation(self):
        # Spec 4.1: epochs, optimiser, input resolution and RGB augmentation stay matched
        students = {name: parse_config(name) for name in STUDENT_CONFIGS}
        for setting in ("epochs", "batch_size", "seed", "no_augment"):
            with self.subTest(setting=setting):
                self.assertEqual(len({getattr(args, setting) for args in students.values()}), 1)
        distillation = [students[name] for name in DISTILLATION_CONFIGS]
        self.assertEqual(len({(args.teacher_checkpoint, args.lambda_distill) for args in distillation}), 1)
        self.assertEqual([args.distill_mode for args in distillation], ["unweighted", "object", "object-quality"])

    def test_only_rgb_augmented_trains_on_degraded_images_and_only_the_teacher_drops_rgb(self):
        # Design A: the teacher and the distilled students train on undegraded images; the teacher
        # alone uses modality dropout, and no student ever has its RGB blanked
        students = {name: parse_config(name) for name in STUDENT_CONFIGS}
        teacher = parse_config("teacher")
        self.assertEqual([name for name, args in students.items() if args.degradation_augmentation], ["rgb_augmented"])
        self.assertFalse(teacher.degradation_augmentation)
        self.assertEqual(teacher.rgb_dropout, 0.15)
        self.assertTrue(all(args.rgb_dropout == 0 for args in students.values()))

    def test_command_line_flags_override_the_config(self):
        args = parse_config("proposed_event", "--epochs", "3", "--lambda-distill", "0.5")
        self.assertEqual((args.epochs, args.lambda_distill, args.distill_mode), (3, 0.5, "object-quality"))

    def test_recorded_command_reproduces_the_run_without_the_config_file(self):
        args = parse_config("rgb_augmented", "--num-workers", "4")
        self.assertNotIn("--config", args.command)
        replayed = parse(*args.command[1:])
        ignored = ("config", "command")
        self.assertEqual(
            {k: v for k, v in vars(replayed).items() if k not in ignored},
            {k: v for k, v in vars(args).items() if k not in ignored},
        )

    def test_unknown_or_misspelt_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for content in ("epoch: 50\n", "resume: runs/student/exp_001\n", "distill_mode: weighted\n"):
                with self.subTest(content=content):
                    config = Path(directory) / "bad.yaml"
                    config.write_text(content, encoding="utf-8")
                    self.assertTrue(rejected("--config", str(config), "--dataset-path", "fake"))
        self.assertTrue(rejected("--config", "configs/does_not_exist.yaml", "--dataset-path", "fake"))

    def test_config_cannot_be_combined_with_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(rejected("--resume", directory, "--config", str(CONFIGS / "rgb_baseline.yaml")))

    def test_a_run_records_the_expanded_command_and_its_config(self):
        loader = MagicMock()
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            training.run,
            select_device=MagicMock(return_value=torch.device("cpu")),
            create_dataloaders=MagicMock(return_value=(loader, loader)),
            extract_trainable_model=MagicMock(),
            create_optimiser=MagicMock(),
            train_model=MagicMock(return_value=Path("best.pt")),
        ), patch.object(training.run.log, "create_log_directory", return_value=Path(directory)), \
                patch.object(training.run.log, "create_logger", return_value=MagicMock()), \
                patch.object(training.run.log, "log_command") as log_command, \
                patch.object(training.run.log, "log_model_config") as log_config, \
                patch("sys.argv", ["train_rgb_student.py", "--config", str(CONFIGS / "rgb_baseline.yaml"),
                                   "--dataset-path", "fake", "--skip-test"]):
            training.main(role="student", rgb_only=True)

        command = log_command.call_args.args[0]
        self.assertEqual(command[0], "train_rgb_student.py")
        self.assertNotIn("--config", command)
        self.assertEqual(command[command.index("--epochs") + 1], str(settings("rgb_baseline")["epochs"]))
        self.assertEqual(log_config.call_args.args[0]["Config File"], str(CONFIGS / "rgb_baseline.yaml"))

    def test_distillation_trains_on_undegraded_images_with_rgb_intact(self):
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
        ), patch.object(training.run.log, "create_log_directory", return_value=Path(directory)),                 patch.object(training.run.log, "create_logger", return_value=MagicMock()),                 patch.object(training.run.log, "log_command"),                 patch.object(training.run.log, "log_model_config"),                 patch("sys.argv", ["train_distilled_student.py", "--config", str(CONFIGS / "proposed_event.yaml"),
                                   "--dataset-path", "fake", "--skip-test"]):
            training.main(role="distill", rgb_only=True)

        student_call, teacher_call = create_loaders.call_args_list
        self.assertFalse(teacher_call.kwargs["rgb_only"])
        for call in (student_call, teacher_call):
            self.assertIsNone(call.kwargs["augmentation_shares"])
            self.assertEqual(call.kwargs.get("rgb_dropout", 0), 0)

if __name__ == "__main__":
    unittest.main()
