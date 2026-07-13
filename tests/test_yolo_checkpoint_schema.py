"""Schema tests for the YOLO checkpoint validator (vision/validate_yolo_checkpoint.py).

GPU-free and does NOT run inference or require a real .pt file -- it exercises the
pure schema-checking logic against synthetic `model.names` dicts, so it verifies
the acceptance/rejection rules deterministically.

Run (project uses unittest):
    python -m unittest discover -s tests -p "test_yolo_checkpoint_schema.py"
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import EXPECTED_YOLO_CLASS_NAMES  # noqa: E402
from vision.validate_yolo_checkpoint import EXPECTED_SCHEMA, check_schema  # noqa: E402

EXPECTED_8 = {
    0: "nut_butter_jar", 1: "whole_nuts", 2: "hand", 3: "cutlery",
    4: "chopping_board", 5: "plate", 6: "bowl", 7: "bread",
}


class TestExpectedSchemaConstant(unittest.TestCase):
    def test_expected_schema_is_exactly_the_8_class_schema(self):
        self.assertEqual(EXPECTED_SCHEMA, EXPECTED_8)
        self.assertEqual({int(k): v for k, v in EXPECTED_YOLO_CLASS_NAMES.items()}, EXPECTED_8)
        self.assertEqual(len(EXPECTED_SCHEMA), 8)

    def test_counter_absent(self):
        self.assertNotIn("counter", EXPECTED_SCHEMA.values())

    def test_bread_is_local_id_7(self):
        self.assertEqual(EXPECTED_SCHEMA[7], "bread")


class TestCheckSchemaAcceptsCorrect(unittest.TestCase):
    def test_correct_schema_passes_all_checks(self):
        ok, names, checks = check_schema(dict(EXPECTED_8))
        self.assertTrue(ok)
        self.assertEqual(names, EXPECTED_8)
        self.assertTrue(all(passed for _, passed, _ in checks))

    def test_correct_schema_accepts_list_form(self):
        # ultralytics may expose names as a list; order 0..7 must still pass.
        as_list = [EXPECTED_8[i] for i in range(8)]
        ok, names, _ = check_schema(as_list)
        self.assertTrue(ok)
        self.assertEqual(names, EXPECTED_8)


class TestCheckSchemaRejects(unittest.TestCase):
    def _failed_labels(self, model_names):
        ok, _, checks = check_schema(model_names)
        self.assertFalse(ok)
        return {label for label, passed, _ in checks if not passed}

    def test_rejects_single_class_cutlery_model(self):
        failed = self._failed_labels({0: "cutlery"})
        self.assertTrue(any("single-class" in lbl for lbl in failed))
        self.assertTrue(any("exact schema match" in lbl for lbl in failed))

    def test_rejects_9_class_schema_with_counter(self):
        nine = {
            0: "nut_butter_jar", 1: "whole_nuts", 2: "hand", 3: "cutlery",
            4: "chopping_board", 5: "plate", 6: "bowl", 7: "counter", 8: "bread",
        }
        failed = self._failed_labels(nine)
        self.assertTrue(any("exactly 8 classes" in lbl for lbl in failed))
        self.assertTrue(any("counter" in lbl for lbl in failed))
        self.assertTrue(any("bread is local id 7" in lbl for lbl in failed))  # id 7 is counter here

    def test_rejects_bread_not_id_7(self):
        # 8 classes but wrong order: bread not at id 7.
        wrong = {
            0: "nut_butter_jar", 1: "whole_nuts", 2: "hand", 3: "cutlery",
            4: "chopping_board", 5: "plate", 6: "bread", 7: "bowl",
        }
        failed = self._failed_labels(wrong)
        self.assertTrue(any("bread is local id 7" in lbl for lbl in failed))
        self.assertTrue(any("exact schema match" in lbl for lbl in failed))

    def test_rejects_renamed_classes(self):
        renamed = dict(EXPECTED_8)
        renamed[0] = "peanut_jar"  # right count/order, wrong name
        failed = self._failed_labels(renamed)
        self.assertTrue(any("exact schema match" in lbl for lbl in failed))


if __name__ == "__main__":
    unittest.main()
