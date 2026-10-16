"""Command-line arguments for a training run, including resuming an interrupted one."""
import argparse
import sys
from pathlib import Path

import log
from dataset_loader import (
    DEFAULT_DEGRADATION_SHARES, DEFAULT_STRENGTH_RANGE, DEFAULT_AUGMENTATION_SEED,
    validate_degradation_shares, validate_strength_range, validate_augmentation_seed,
)
from distillation import DISTILL_MODES

# Settings that may change when resuming: they affect speed and logging, not the training itself
RESUME_OVERRIDES = ("num_workers", "console_level")


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
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
        help="Randomly mix clean, dark and overexposed images for RGB-only student training.",
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
    args = parser.parse_args()
    if args.resume is not None:
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
    if args.distill_mode != "unweighted" and args.teacher_checkpoint is None:
        parser.error("--distill-mode needs --teacher-checkpoint")
    return args


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
