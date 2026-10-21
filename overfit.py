"""Overfit the RGB student on a tiny training subset to check the pipeline can memorise frames.

Usage: python overfit.py --dataset-path DATASET --epochs 50 [--max-samples 8]
"""
from training.overfit import main


if __name__ == "__main__":
    main()
