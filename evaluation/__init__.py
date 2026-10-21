"""Evaluation of trained detectors, used both during training and on its own.

    metrics               validation, test and degraded-benchmark metrics (precision, recall, mAP)
    benchmark_reporting   traceable JSON exports and comparisons of benchmark results
    cli                   main() for evaluate.py

Evaluation never imports the training package, so the RGB-only evaluation path stays free of the
teacher, the object mask and the other training-only components.
"""
