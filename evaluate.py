"""Evaluate a trained student or teacher on the test split, or on the clean/dark/overexposed benchmark.

Usage: python evaluate.py --weights PATH --type student --dataset-path DATASET [--benchmark]
"""
from evaluation.cli import main


if __name__ == "__main__":
    main()
