"""Traceable exports and compatible cross-model comparisons for the DSEC benchmark."""

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = ("clean", "dark", "overexposed")
EXPECTED_MODELS = (
    "rgb_baseline",
    "rgb_augmented",
    "naive_distillation",
    "object_distillation",
    "proposed_event",
    "teacher",
)
METRICS = ("map50", "map50_95", "recall", "map50_drop", "map50_95_drop", "recall_drop")
REPORT_VERSION = 1


def _git_output(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(
    results: dict,
    *,
    model_name: str,
    model_type: str,
    model,
    checkpoint_path: Path,
    dataset_path: Path,
    dataset_id: str,
    degradation_strength: float,
    evaluation_seed: int,
    max_samples: int | None,
    batch_size: int,
    num_workers: int,
    code_root: Path | None = None,
) -> dict:
    """Record evaluation inputs and observed metrics without changing the evaluator."""
    checkpoint_path = checkpoint_path.absolute()
    root = code_root or Path(__file__).resolve().parents[1]
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    return {
        "format_version": REPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_type": model_type,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "code_commit": commit,
        "code_dirty": None if status is None else bool(status),
        "dataset_path": str(dataset_path.absolute()),
        "dataset_id": dataset_id,
        "split": "test",
        "degradation_strength": float(degradation_strength),
        "evaluation_seed": evaluation_seed,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "conditions": results,
    }


def save_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError(f"{path}: benchmark report must be a JSON object")
    if report.get("format_version") != REPORT_VERSION:
        raise ValueError(f"{path}: unsupported benchmark report format")
    for key in ("model_name", "model_type", "checkpoint_sha256", "dataset_id", "code_commit"):
        if not report.get(key):
            raise ValueError(f"{path}: missing {key}")
    if report.get("code_dirty") is not False:
        raise ValueError(f"{path}: evaluation code has uncommitted changes or unknown Git state")
    for condition in CONDITIONS:
        row = report.get("conditions", {}).get(condition)
        if not isinstance(row, dict) or not row.get("fingerprint"):
            raise ValueError(f"{path}: missing {condition} input fingerprint")
        for metric in METRICS:
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{path}: invalid {condition} {metric}")
    return report


def compare_reports(paths: list[Path]) -> dict:
    """Reject incomparable results; retain explicit blanks for missing milestone models."""
    if not paths:
        raise ValueError("Provide at least one benchmark report")
    reports = [load_report(path) for path in paths]
    first = reports[0]
    names = set()
    checkpoints = set()
    settings = ("dataset_id", "split", "degradation_strength", "evaluation_seed", "max_samples", "code_commit")
    for report, path in zip(reports, paths):
        name = report["model_name"]
        if name in names:
            raise ValueError(f"Duplicate model name: {name}")
        names.add(name)
        checksum = report["checkpoint_sha256"]
        if checksum in checkpoints:
            raise ValueError(f"{path}: checkpoint already used under another model name")
        checkpoints.add(checksum)
        for setting in settings:
            if report.get(setting) != first.get(setting):
                raise ValueError(f"{path}: {setting} differs from the first report")
        for condition in CONDITIONS:
            if report["conditions"][condition]["fingerprint"] != first["conditions"][condition]["fingerprint"]:
                raise ValueError(f"{path}: {condition} RGB inputs differ from the first report")

    rows = []
    for name in (*EXPECTED_MODELS, *(sorted(names - set(EXPECTED_MODELS)))):
        match = next((report for report in reports if report["model_name"] == name), None)
        for condition in CONDITIONS:
            row = {"model_name": name, "condition": condition, "status": "measured" if match else "missing"}
            for metric in METRICS:
                row[metric] = match["conditions"][condition][metric] if match else None
            rows.append(row)
    return {
        "format_version": REPORT_VERSION,
        "comparison_settings": {setting: first.get(setting) for setting in settings},
        "missing_models": [name for name in EXPECTED_MODELS if name not in names],
        "reports": [str(path.absolute()) for path in paths],
        "rows": rows,
    }


def save_comparison(directory: Path, comparison: dict) -> tuple[Path, Path]:
    json_path = directory / "comparison.json"
    csv_path = directory / "comparison.csv"
    for path in (json_path, csv_path):
        if path.exists():
            raise FileExistsError(f"Comparison output already exists: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    save_json(json_path, comparison)
    columns = ("model_name", "condition", "status", *METRICS)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(comparison["rows"])
    return json_path, csv_path
