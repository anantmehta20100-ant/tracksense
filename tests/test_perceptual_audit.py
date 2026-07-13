import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ml.perceptual_audit import audit_dataset, parse_args


class PerceptualAuditTests(unittest.TestCase):
    def test_explicit_root_and_report_write_cross_split_review_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "candidate"
            for split in ("train", "valid", "test"):
                (root / split / "images").mkdir(parents=True)

            for split in ("train", "valid"):
                Image.new("RGB", (32, 32), "white").save(
                    root / split / "images" / f"cutlery__{split}.jpg"
                )

            report_path = Path(tmp) / "near_duplicate_review_final.csv"
            summary = audit_dataset(root, report_path)

            self.assertEqual(summary["total_suspicious_pairs"], 1)
            self.assertEqual(summary["cross_split_suspicious_pairs"], 1)
            self.assertEqual(summary["strongest_matches"], 1)

            with report_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["path_a"], "train/images/cutlery__train.jpg")
            self.assertEqual(rows[0]["path_b"], "valid/images/cutlery__valid.jpg")
            self.assertEqual(rows[0]["split_a"], "train")
            self.assertEqual(rows[0]["split_b"], "valid")
            self.assertEqual(rows[0]["class_name"], "cutlery")
            self.assertEqual(rows[0]["hash_distance"], "0")
            self.assertEqual(rows[0]["review_priority"], "strongest")

    def test_explicit_cli_paths_override_defaults(self):
        args = parse_args([
            "--root", "data/training_8class_balanced",
            "--report", "reports/near_duplicate_review_final.csv",
        ])

        self.assertEqual(args.root, "data/training_8class_balanced")
        self.assertEqual(args.report, "reports/near_duplicate_review_final.csv")


if __name__ == "__main__":
    unittest.main()
