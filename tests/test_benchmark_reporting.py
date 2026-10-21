import csv
import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from evaluation.benchmark_reporting import build_report, compare_reports, save_comparison, save_json


def sample_report(name: str, checksum: str) -> dict:
    conditions = {}
    for condition in ("clean", "dark", "overexposed"):
        conditions[condition] = {
            "map50": 0.6,
            "map50_95": 0.4,
            "recall": 0.7,
            "map50_drop": 0.0,
            "map50_95_drop": 0.0,
            "recall_drop": 0.0,
            "fingerprint": f"{condition}-same-inputs",
        }
    return {
        "format_version": 1,
        "model_name": name,
        "model_type": "student",
        "checkpoint_sha256": checksum,
        "dataset_id": "agreed-dsec-release",
        "split": "test",
        "code_commit": "abc123",
        "code_dirty": False,
        "degradation_strength": 1.0,
        "evaluation_seed": 42,
        "max_samples": None,
        "conditions": conditions,
    }


class BenchmarkReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0

    def write(self, name: str, report: dict) -> Path:
        self.serial += 1
        path = self.root / f"{name}-{self.serial}.json"
        save_json(path, report)
        return path

    def test_export_records_checkpoint_identity_and_provenance(self):
        checkpoint = self.root / "best.pt"
        checkpoint.write_bytes(b"checkpoint-data")

        class Parameter:
            def numel(self):
                return 7

        class Model:
            def parameters(self):
                return [Parameter(), Parameter()]

        with patch("evaluation.benchmark_reporting._git_output", side_effect=["abc123", ""]):
            report = build_report(
                sample_report("rgb_baseline", "unused")["conditions"],
                model_name="rgb_baseline",
                model_type="student",
                model=Model(),
                checkpoint_path=checkpoint,
                dataset_path=self.root,
                dataset_id="agreed-dsec-release",
                degradation_strength=1.0,
                evaluation_seed=42,
                max_samples=None,
                batch_size=16,
                num_workers=0,
            )

        self.assertEqual(report["checkpoint_sha256"], hashlib.sha256(b"checkpoint-data").hexdigest())
        self.assertEqual(report["parameter_count"], 14)
        self.assertEqual(report["code_commit"], "abc123")
        self.assertIs(report["code_dirty"], False)

    def test_compatible_reports_export_measured_and_missing_rows(self):
        first = self.write("baseline", sample_report("rgb_baseline", "hash-1"))
        second = self.write("augmented", sample_report("rgb_augmented", "hash-2"))
        comparison = compare_reports([first, second])
        json_path, csv_path = save_comparison(self.root / "comparison", comparison)

        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["missing_models"], [
            "naive_distillation", "object_distillation", "proposed_event", "teacher"
        ])
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 18)  # 6 models x 3 conditions
        self.assertEqual(rows[0]["status"], "measured")
        self.assertEqual(rows[0]["map50"], "0.6")
        self.assertEqual(rows[-1]["status"], "missing")
        self.assertEqual(rows[-1]["map50"], "")

    def test_rejects_mismatched_settings_and_inputs(self):
        first = self.write("baseline", sample_report("rgb_baseline", "hash-1"))
        base = sample_report("rgb_augmented", "hash-2")
        for changed, value in (
            ("evaluation_seed", 43),
            ("dataset_id", "different-labels"),
            ("code_commit", "different-code"),
        ):
            with self.subTest(changed=changed):
                candidate = deepcopy(base)
                candidate[changed] = value
                second = self.write("augmented", candidate)
                with self.assertRaisesRegex(ValueError, changed):
                    compare_reports([first, second])

        candidate = deepcopy(base)
        candidate["conditions"]["dark"]["fingerprint"] = "different-images"
        second = self.write("augmented", candidate)
        with self.assertRaisesRegex(ValueError, "dark RGB inputs"):
            compare_reports([first, second])

    def test_rejects_dirty_code_and_duplicate_checkpoint(self):
        first = self.write("baseline", sample_report("rgb_baseline", "hash-1"))
        candidate = sample_report("rgb_augmented", "hash-2")
        candidate["code_dirty"] = True
        second = self.write("augmented", candidate)
        with self.assertRaisesRegex(ValueError, "uncommitted changes"):
            compare_reports([first, second])

        candidate["code_dirty"] = False
        candidate["checkpoint_sha256"] = "hash-1"
        second = self.write("augmented", candidate)
        with self.assertRaisesRegex(ValueError, "checkpoint already used"):
            compare_reports([first, second])

    def test_outputs_never_overwrite_existing_files(self):
        path = self.write("baseline", sample_report("rgb_baseline", "hash-1"))
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_json(path, {"changed": True})
        self.assertEqual(path.read_bytes(), original)

        comparison = compare_reports([path])
        directory = self.root / "comparison"
        json_path, csv_path = save_comparison(directory, comparison)
        original_json = json_path.read_bytes()
        original_csv = csv_path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_comparison(directory, comparison)
        self.assertEqual(json_path.read_bytes(), original_json)
        self.assertEqual(csv_path.read_bytes(), original_csv)


if __name__ == "__main__":
    unittest.main()
