"""Combine compatible clean/dark/overexposed benchmark reports."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.benchmark_reporting import compare_reports, save_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Reports exported by evaluate.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        comparison = compare_reports(args.reports)
        json_path, csv_path = save_comparison(args.output_dir, comparison)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Saved {json_path} and {csv_path}")
    if comparison["missing_models"]:
        print("Missing models: " + ", ".join(comparison["missing_models"]))


if __name__ == "__main__":
    main()
