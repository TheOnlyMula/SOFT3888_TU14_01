"""Command line for evaluating a trained student or teacher: test metrics or the degraded benchmark.

Usage: python evaluate.py --weights PATH --type student --dataset-path DATASET [--benchmark]
"""
import argparse
import sys
from pathlib import Path

from loaders.dataset_loader import create_test_loader
from utils import log
from utils.model import extract_trainable_model, select_device

from .benchmark_reporting import build_report, save_json
from .metrics import (
    degradation_strength_argument,
    evaluate_degraded_benchmark,
    evaluate_test_model,
    format_benchmark_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to the trained weights (.pt)"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["student", "teacher"],
        help="Evaluating student or teacher",
    )
    parser.add_argument(
        "--dataset-path", type=Path, required=True, help="Path to the dataset directory"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Evaluate clean, dark and overexposed benchmark conditions.",
    )
    parser.add_argument(
        "--degradation-strength",
        type=degradation_strength_argument,
        default=1.0,
        help='Degradation strength: "mild", "strong", or a number in [0, 1].',
    )
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument(
        "--benchmark-report", type=Path,
        help="Save benchmark metrics and provenance as JSON at this path.",
    )
    parser.add_argument(
        "--model-name",
        help="Model label for a benchmark report, e.g. rgb_baseline or naive_distillation.",
    )
    parser.add_argument(
        "--dataset-id",
        help="Shared identifier for the exact DSEC image and annotation release.",
    )
    args = parser.parse_args()
    if args.benchmark_report and not args.benchmark:
        parser.error("--benchmark-report requires --benchmark")
    if args.benchmark_report and (not args.model_name or not args.dataset_id or not args.dataset_id.strip()):
        parser.error("--benchmark-report requires --model-name and --dataset-id")
    if args.benchmark_report and (args.model_name == "teacher") != (args.type == "teacher"):
        parser.error("--model-name teacher must match --type teacher")

    log_directory = log.create_log_directory(args.type)
    logger = log.create_logger("evaluate", log_directory)
    log.log_command(sys.argv, log_directory)
    device = select_device()

    rgb_only = args.type == "student"
    model = extract_trainable_model(
        ch=3 if rgb_only else 5,
        weights_path=None,
        device=device,
    )

    if args.benchmark:
        benchmark_results = evaluate_degraded_benchmark(
            dataset_path=args.dataset_path,
            model=model,
            device=device,
            logger=logger,
            checkpoint_path=args.weights,
            rgb_only=rgb_only,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            degradation_strength=args.degradation_strength,
            evaluation_seed=args.evaluation_seed,
        )

        for line in format_benchmark_table(benchmark_results):
            logger.info(line)

        for condition, result in benchmark_results.items():
            logger.info(
                "%s fingerprint: %s",
                condition,
                result["fingerprint"],
            )
        if args.benchmark_report:
            report = build_report(
                benchmark_results,
                model_name=args.model_name,
                model_type=args.type,
                model=model,
                checkpoint_path=args.weights,
                dataset_path=args.dataset_path,
                dataset_id=args.dataset_id.strip(),
                degradation_strength=args.degradation_strength,
                evaluation_seed=args.evaluation_seed,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            save_json(args.benchmark_report, report)
            logger.info("Benchmark report saved to %s", args.benchmark_report)

    else:
        test_loader = create_test_loader(
            args.dataset_path,
            logger,
            rgb_only=rgb_only,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )

        evaluate_test_model(
            test_loader,
            model,
            device,
            logger,
            args.weights,
            save_dir=log_directory / "test",
        )
