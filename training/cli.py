"""Command-line arguments for a training run, from a config file and flags, including resuming one."""
import argparse
import sys
from pathlib import Path

import yaml

from utils import log
from degradation.conditions import (
    DEFAULT_DEGRADATION_SHARES, DEFAULT_STRENGTH_RANGE, DEFAULT_AUGMENTATION_SEED,
    validate_degradation_shares, validate_strength_range, validate_augmentation_seed,
)
from .distillation import DISTILL_MODES

# Settings that may change when resuming: they affect speed and logging, not the training itself
RESUME_OVERRIDES = ("num_workers", "console_level")


def parse_args(description: str) -> argparse.Namespace:
    """Parse a training command, with any --config file's settings applied before its flags.

    The returned settings also hold `config` (the file, or None) and `command`: the full command
    with the file's settings spelled out as flags, which the run records so it can be resumed
    without the file.
    """
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML file of an experiment's settings (see configs/), keyed by these options' names "
             "with underscores. Options given on the command line override it.",
    )
    parser.add_argument(
        "--dataset-path", type=Path, default=None,
        help="DSEC root with labelled train and val directories. Required unless resuming.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, metavar="RUN_DIR",
        help="Continue an interrupted run (e.g. runs/student/exp_006) from its latest checkpoint. "
             "All settings come from the original run; only --num-workers and --console-level "
             "may be given again.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1, help="Total number of epochs to train.")
    parser.add_argument(
        "--degradation-augmentation", action="store_true",
        help="Randomly mix clean, dark and overexposed training images. With events (teacher, "
             "distillation) only the RGB channels are degraded; the events stay clean.",
    )
    parser.add_argument(
        "--augmentation-seed", type=int, default=DEFAULT_AUGMENTATION_SEED,
        help="Non-negative seed for training augmentation only (default: 0); independent of evaluation.",
    )
    parser.add_argument(
        "--degradation-shares", type=float, nargs=3,
        metavar=("CLEAN", "DARK", "OVEREXPOSED"), default=DEFAULT_DEGRADATION_SHARES,
        help="Condition probabilities summing to 1 (default: 0.5 0.25 0.25).",
    )
    parser.add_argument(
        "--dark-strength-range", type=float, nargs=2, metavar=("MIN", "MAX"),
        default=DEFAULT_STRENGTH_RANGE,
        help="Uniform training darkening strength range in [0, 1] (default: 0.35 1.0).",
    )
    parser.add_argument(
        "--overexposed-strength-range", type=float, nargs=2, metavar=("MIN", "MAX"),
        default=DEFAULT_STRENGTH_RANGE,
        help="Uniform training over-exposure strength range in [0, 1] (default: 0.35 1.0).",
    )
    parser.add_argument(
        "--rgb-dropout", type=float, default=0.0, metavar="P",
        help="Teacher only (modality dropout): blank the RGB of this share of training images, "
             "keeping their events and labels, so the teacher also learns to detect from events.",
    )
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Run training and validation only; do not run final eval step.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit each of train/val/test to this many evenly spaced frames (for quick smoke runs).",
    )
    parser.add_argument(
        "--teacher-checkpoint", type=Path, default=None,
        help="Frozen RGB + event teacher to distil from. Training then uses paired RGB + event "
             "batches, while the student, validation and test stay RGB-only.",
    )
    parser.add_argument(
        "--lambda-distill", type=float, default=1.0,
        help="Weight of the distillation term added to the detection loss.",
    )
    parser.add_argument(
        "--distill-mode", choices=DISTILL_MODES, default="unweighted",
        help="Where the distillation term compares features: 'unweighted' everywhere (baseline), "
             "'object' only inside the enlarged labelled boxes, or 'object-quality' as 'object' "
             "with each image's loss scaled by w = clip(s_e * c_t, 0, 1). Needs --teacher-checkpoint.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for model initialisation, shuffling and augmentation. Use the same seed for "
             "every student you compare.",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Train on unaugmented frames, e.g. for the tiny-subset overfit test.",
    )
    parser.add_argument(
        "--console-level", type = str, default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help = "Set the logging level.",
    )
    config, command = expand_config(parser, sys.argv[1:])
    args = parser.parse_args(command)
    if args.resume is not None:
        if config is not None:
            parser.error("--resume takes its settings from the original run, so --config cannot be given")
        args = resumed_args(parser, args)
    elif args.dataset_path is None:
        parser.error("--dataset-path is required unless resuming")
    try:
        args.augmentation_seed = validate_augmentation_seed(args.augmentation_seed)
        args.degradation_shares = validate_degradation_shares(args.degradation_shares)
        args.dark_strength_range = validate_strength_range(args.dark_strength_range)
        args.overexposed_strength_range = validate_strength_range(args.overexposed_strength_range)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.lambda_distill < 0:
        parser.error("--lambda-distill must be non-negative")
    if not 0 <= args.rgb_dropout < 1:
        parser.error("--rgb-dropout must be a probability in [0, 1)")
    if args.distill_mode != "unweighted" and args.teacher_checkpoint is None:
        parser.error("--distill-mode needs --teacher-checkpoint")
    args.config = config
    args.command = [sys.argv[0], *command]
    return args


def expand_config(parser: argparse.ArgumentParser, argv: list[str]) -> tuple[Path | None, list[str]]:
    config_option = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_option.add_argument("--config", type=Path)
    known, rest = config_option.parse_known_args(argv)
    if known.config is None:
        return None, argv
    if not known.config.is_file():
        parser.error(f"--config: file not found: {known.config}")
    with open(known.config, encoding="utf-8") as file:
        settings = yaml.safe_load(file) or {}

    options = []
    for name, value in settings.items():
        if name in ("config", "resume"):
            parser.error(f"{known.config}: '{name}' cannot be set in a config file")
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            options += [flag] if value else []
        elif isinstance(value, list):
            options += [flag, *map(str, value)]
        else:
            options += [flag, str(value)]
    return known.config, options + rest


def resumed_args(parser: argparse.ArgumentParser, given: argparse.Namespace) -> argparse.Namespace:
    """Rebuild a run's original settings from its folder, keeping the allowed overrides."""
    run_directory = given.resume
    if not run_directory.is_dir():
        parser.error(f"--resume: run folder not found: {run_directory}")

    changed = [
        "--" + name.replace("_", "-")
        for name, value in vars(given).items()
        if name not in ("resume", *RESUME_OVERRIDES) and value != parser.get_default(name)
    ]
    if changed:
        parser.error(
            f"--resume takes its settings from the original run, so {', '.join(changed)} cannot "
            "be changed. Start a new run instead."
        )

    original_command = log.load_command(run_directory)
    script = Path(sys.argv[0]).name
    if Path(original_command[0]).name != script:
        parser.error(
            f"--resume: {run_directory} was started with {Path(original_command[0]).name}, "
            f"not {script}"
        )

    args = parser.parse_args(original_command[1:])
    args.resume = run_directory
    for name in RESUME_OVERRIDES:
        if getattr(given, name) != parser.get_default(name):
            setattr(args, name, getattr(given, name))
    return args
