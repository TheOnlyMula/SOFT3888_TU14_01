"""Measure the event-histogram scale constant K used to normalise the teacher's event channels.

K is a percentile of non-zero per-pixel event counts, measured on the TRAINING split only
(never val or test), so the constant carries no information about the evaluation data.

Usage:
    python tools/measure_event_scale.py --dataset-path dataset                  # 3 frames/sequence
    python tools/measure_event_scale.py --dataset-path dataset --all-frames     # every frame
    python tools/measure_event_scale.py --dataset-path dataset --all-frames --workers 8
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loaders.dataset_loader import DSECFusionDataset, load_causal_events 

MAX_COUNT = 4096

_worker_dataset = None


def event_counts(base_dataset, idx: int) -> np.ndarray:
    """Per-pixel event counts for one frame, both polarities, before any scaling."""
    events = load_causal_events(base_dataset, idx)
    histogram = np.zeros((2, 480, 640), dtype=np.int64)
    x = events["x"].astype(int)
    y = events["y"].astype(int)
    p = events["p"].astype(int)
    np.add.at(histogram[0], (y[p == 0], x[p == 0]), 1)
    np.add.at(histogram[1], (y[p == 1], x[p == 1]), 1)
    return histogram


def frame_tally(base_dataset, idx: int) -> np.ndarray:
    """How many pixels had each event count, for one frame."""
    counts = np.clip(event_counts(base_dataset, idx).ravel(), 0, MAX_COUNT - 1)
    return np.bincount(counts, minlength=MAX_COUNT)


def sample_indices(base_dataset, frames_per_sequence: int | None) -> list[int]:
    """Frame indices to read: every frame, or evenly spaced ones within each sequence."""
    indices, start = [], 0
    for directory in base_dataset.subsequence_directories:
        count = len(base_dataset.img_idx_track_idxs[directory.name]) - 1
        if frames_per_sequence is None:
            indices.extend(range(start, start + count))
        else:
            step = max(1, count // (frames_per_sequence + 1))
            indices.extend(start + min(count - 1, step * (i + 1)) for i in range(frames_per_sequence))
        start += count
    return indices


def percentile_from_tally(tally: np.ndarray, percentile: float) -> int:
    """Smallest count at or below which `percentile`% of non-zero pixels fall."""
    non_zero = tally.copy()
    non_zero[0] = 0
    total = non_zero.sum()
    if total == 0:
        raise ValueError("No events were found; cannot measure a scale constant.")
    target = percentile / 100.0 * total
    return int(np.searchsorted(np.cumsum(non_zero), target))


def _init_worker(dataset_path: str, split: str) -> None:
    global _worker_dataset
    _worker_dataset = DSECFusionDataset(Path(dataset_path), split=split, rgb_only=False)


def _worker_tally(idx: int) -> np.ndarray:
    return frame_tally(_worker_dataset.base_dataset, idx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                        help="Measure on train only; the others are for inspection.")
    parser.add_argument("--frames-per-sequence", type=int, default=3)
    parser.add_argument("--all-frames", action="store_true", help="Read every frame in the split.")
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--workers", type=int, default=0, help="Processes to read frames with.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file for the result.")
    args = parser.parse_args()

    if args.split != "train":
        print(f"WARNING: measuring on '{args.split}'. Only the train split may set the frozen constant.")

    dataset = DSECFusionDataset(args.dataset_path, split=args.split, rgb_only=False)
    base = dataset.base_dataset
    indices = sample_indices(base, None if args.all_frames else args.frames_per_sequence)
    sequences = len(base.subsequence_directories)
    print(f"Reading {len(indices)} frames from {sequences} '{args.split}' sequences", flush=True)

    tally = np.zeros(MAX_COUNT, dtype=np.int64)
    start_time = time.monotonic()

    def report(done: int) -> None:
        elapsed = time.monotonic() - start_time
        rate = done / elapsed if elapsed else 0.0
        remaining = (len(indices) - done) / rate if rate else 0.0
        print(f"  {done}/{len(indices)} frames | {rate:.1f} frames/s | eta {remaining / 60:.1f} min", flush=True)

    if args.workers > 0:
        from multiprocessing import Pool

        with Pool(args.workers, initializer=_init_worker,
                  initargs=(str(args.dataset_path), args.split)) as pool:
            for done, result in enumerate(
                pool.imap_unordered(_worker_tally, indices, chunksize=8), start=1
            ):
                tally += result
                if done % 500 == 0 or done == len(indices):
                    report(done)
    else:
        for done, idx in enumerate(indices, start=1):
            tally += frame_tally(base, idx)
            if done % 100 == 0 or done == len(indices):
                report(done)

    reported = [50, 90, 99, 99.9, args.percentile]
    values = {p: percentile_from_tally(tally, p) for p in reported}
    highest = int(np.nonzero(tally)[0].max())
    non_zero_pixels = int(tally[1:].sum())

    print("\nNon-zero per-pixel event counts:")
    for p in sorted(set(reported)):
        print(f"  p{p:<6} {values[p]:8d}")
    print(f"  max     {highest:8d}")
    print(f"  pixels with events: {non_zero_pixels:,} of {int(tally.sum()):,}")
    print(f"\nEVENT_SCALE_K (p{args.percentile}) = {values[args.percentile]}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": args.split,
            "sequences": sequences,
            "frames_read": len(indices),
            "all_frames": bool(args.all_frames),
            "percentile": args.percentile,
            "event_scale_k": values[args.percentile],
            "percentiles": {str(p): values[p] for p in sorted(set(reported))},
            "max_count": highest,
            "non_zero_pixels": non_zero_pixels,
            "total_pixels": int(tally.sum()),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
