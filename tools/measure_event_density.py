"""Measure and freeze whole-frame event coverage on the training split only.

Usage: python tools/measure_event_density.py --dataset-path DSEC
The reference is the 90th percentile of frame coverage, including empty frames.
Every training frame is read, without augmentation; all configured training
sequences must be available. Existing outputs
are never replaced; the saved reference is reused by training, not remeasured.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_loader import DSECFusionDataset, EVENT_WINDOW_US, SPLIT_CONFIG_PATH, load_causal_events
from dsec_det.io import yaml_file_to_dict


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "event_density_reference.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Frozen reference already exists: {args.output}; refusing to overwrite")

    dataset = DSECFusionDataset(args.dataset_path, split="train", rgb_only=False, augment=False)
    base = dataset.base_dataset
    if len(base) == 0:
        parser.error("No training frames available")
    sequences = [directory.name for directory in base.subsequence_directories]
    configured = yaml_file_to_dict(SPLIT_CONFIG_PATH)["train"]
    missing = sorted(set(configured) - set(sequences))
    if missing:
        parser.error(
            "Complete training split required; missing sequences: " + ", ".join(missing)
        )

    coverage = []
    for idx in range(len(base)):
        histogram = dataset.generate_event_histogram(
            load_causal_events(base, idx), height=base.height, width=base.width
        )
        active = (histogram > 0).any(dim=0)
        coverage.append(int(active.sum()) / (base.height * base.width))
        if (idx + 1) % 100 == 0 or idx + 1 == len(base):
            print(f"Read {idx + 1}/{len(base)} training frames", flush=True)

    reference = float(np.percentile(coverage, 90, method="linear"))
    if reference <= 0:
        parser.error("Training coverage p90 is zero; cannot freeze a positive reference")
    metadata = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "train",
        "region": "whole_frame",
        "event_window_us": EVENT_WINDOW_US,
        "percentile": 90,
        "percentile_method": "linear",
        "event_density_reference": reference,
        "frames_read": len(coverage),
        "all_available_frames": True,
        "sequences": sequences,
        "missing_training_sequences": missing,
        "complete_training_split": not missing,
        "augmentation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    print(f"Frozen reference {reference:.8f} saved to {args.output}")


if __name__ == "__main__":
    main()
