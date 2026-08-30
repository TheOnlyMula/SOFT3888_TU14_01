"""Exercise the real project training/test functions on small disjoint subsets.

Run from the Git checkout with its .venv Python. No project source is modified.
The training/validation pools use the repository's existing seeded split.
This is an integration smoke run, not a full benchmark or an accuracy claim.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    data = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    if not (repo / "train_rgb_student.py").is_file():
        raise FileNotFoundError("Run this command from the project Git checkout.")
    if min(args.train_samples, args.val_samples, args.cpu_threads) < 1:
        raise ValueError("Sample limits and CPU thread count must be positive.")
    if output.exists():
        raise FileExistsError(f"Choose a NEW output folder; previous runs are not overwritten: {output}")
    for split in ("train", "test"):
        if not (data / split).is_dir():
            raise FileNotFoundError(f"Missing data split: {data / split}")
    train_names = {p.name for p in (data / "train").iterdir() if p.is_dir()}
    test_names = {p.name for p in (data / "test").iterdir() if p.is_dir()}
    if train_names & test_names:
        raise ValueError(f"Train/test sequence names overlap: {sorted(train_names & test_names)}")

    sys.path.insert(0, str(repo))
    import torch
    import ultralytics
    from torch.utils.data import DataLoader, Subset
    from train_rgb_student import (
        create_rgb_dataloaders, create_rgb_test_loader,
        extract_trainable_model, create_optimiser,
        train_model, evaluate_test_model,
    )

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    device = torch.device("cpu")
    train_pool, val_pool = create_rgb_dataloaders(data, args.batch_size, 0)
    test_loader = create_rgb_test_loader(data, args.batch_size, 0)
    if len(train_pool.dataset) < args.train_samples or len(val_pool.dataset) < args.val_samples:
        raise ValueError("Not enough training/validation samples for the requested small run.")

    def limited_loader(pool, count, shuffle):
        return DataLoader(
            Subset(pool.dataset, list(range(count))),
            batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
            pin_memory=False, collate_fn=pool.collate_fn,
        )

    train_loader = limited_loader(train_pool, args.train_samples, True)
    val_loader = limited_loader(val_pool, args.val_samples, False)
    train_indices = list(train_pool.dataset.indices[:args.train_samples])
    val_indices = list(val_pool.dataset.indices[:args.val_samples])
    if set(train_indices) & set(val_indices):
        raise AssertionError("Training/validation indices overlap.")
    # Read real batches before downloading weights or starting optimisation.
    for name, loader in (("train", train_loader), ("validation", val_loader), ("test", test_loader)):
        images, targets = next(iter(loader))
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Unexpected {name} image shape: {tuple(images.shape)}")
        print(f"Preflight {name}: images={tuple(images.shape)}, targets={tuple(targets.shape)}", flush=True)

    weights_file = Path(args.weights)
    weights = str(weights_file.resolve()) if weights_file.is_file() else args.weights
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "scope": "small real-model integration run, NOT a full test benchmark",
        "repo_root": str(repo), "dataset_root": str(data),
        "python": sys.version, "torch": str(torch.__version__),
        "ultralytics": str(ultralytics.__version__), "device": "cpu",
        "initial_weights_or_config": weights,
        "batch_size": args.batch_size, "seed": 42,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "train_sequences": sorted(train_names), "test_sequences": sorted(test_names),
        "train_indices_in_base_dataset": train_indices,
        "validation_indices_in_base_dataset": val_indices,
        "test_is_used_only_after_training_for_loss_evaluation": True,
        "note": "Preflight checks file loading only. Team must still confirm split provenance.",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Small run: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test", flush=True)
    print(f"Run output: {output}", flush=True)
    original_cwd = Path.cwd()
    start = time.monotonic()
    try:
        # Any automatic model download stays in this run's output folder.
        os.chdir(output)
        model = extract_trainable_model(weights, device=str(device))
        optimiser = create_optimiser(model)
        checkpoint = train_model(train_loader, val_loader, model, optimiser,
                                 device, output / "best_custom_yolo.pt")
        avg_batch, avg_image = evaluate_test_model(test_loader, model, device, checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored = all(torch.equal(value.detach().cpu(), saved[name])
                       for name, value in model.state_dict().items())
        if not restored:
            raise AssertionError("Evaluated model state does not match the saved best checkpoint.")
        result = {
            "scope": metadata["scope"], "checkpoint": str(checkpoint),
            "best_checkpoint_restored_and_unchanged_during_test": restored,
            "test_samples": len(test_loader.dataset), "test_batches": len(test_loader),
            "average_test_batch_loss": avg_batch, "average_test_per_image_loss": avg_image,
            "elapsed_seconds": time.monotonic() - start,
        }
        (output / "test_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("Best checkpoint verified after evaluation: True", flush=True)
        print(f"Saved measured test results: {output / 'test_results.json'}", flush=True)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
