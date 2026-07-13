"""Headless demo runner tests (no camera, replay-based).

Locks the flagship success criteria that the physical demo will later confirm:
the replay -> contact_tracker -> RF -> RiskEngine flow emits EXACTLY the three
meaningful flagship contacts, propagates risk down the chain to the plate, and
writes the three dashboard-ready artifacts marked physical_verification=false.

A small RF is trained ONCE into a temp dir (setUpModule), so the suite stays fast
and never touches the real model artifact. Tendencies + chain correctness are
asserted, never brittle exact probabilities.

Run:  python -m unittest discover -s tests -p "test_headless_demo.py"
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.generate_risk_training_data import generate_dataset  # noqa: E402
from ml.train_random_forest import parse_args, train  # noqa: E402
from pipeline.headless_demo_runner import FLAGSHIP_PAIRS, run  # noqa: E402

_TMP_DIR = None
_MODEL_PATH = None
_META_PATH = None
_OUT_DIR = None


def setUpModule():
    global _TMP_DIR, _MODEL_PATH, _META_PATH, _OUT_DIR
    _TMP_DIR = tempfile.mkdtemp(prefix="tracksense_headless_")
    csv_path = os.path.join(_TMP_DIR, "risk_events.csv")
    _MODEL_PATH = os.path.join(_TMP_DIR, "rf.joblib")
    _META_PATH = os.path.join(_TMP_DIR, "rf.json")
    _OUT_DIR = os.path.join(_TMP_DIR, "headless_demo")
    events_df, _ = generate_dataset(num_scenarios=400, seed=42)
    events_df.to_csv(csv_path, index=False)
    train(parse_args(["--no-search", "--n-estimators", "80", "--min-samples-leaf", "2",
                      "--data", csv_path, "--model-out", _MODEL_PATH,
                      "--metadata-out", _META_PATH, "--seed", "42"]))


def tearDownModule():
    if _TMP_DIR:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)


class TestHeadlessFlagship(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run("flagship_chain", frames=150, model_path=_MODEL_PATH, out_dir=_OUT_DIR)

    def test_exactly_three_meaningful_contacts(self):
        self.assertEqual(len(self.result["contacts"]), 3)

    def test_the_three_contacts_are_the_flagship_chain(self):
        seen = {frozenset((c["source_class"], c["target_class"])) for c in self.result["contacts"]}
        for a, b in FLAGSHIP_PAIRS:
            self.assertIn(frozenset((a, b)), seen, f"missing contact {a}<->{b}")

    def test_object_object_contact_without_hand(self):
        # None of the flagship contacts involve a hand -- object<->object works.
        for c in self.result["contacts"]:
            self.assertNotIn("hand", (c["source_class"], c["target_class"]))

    def test_plate_has_downstream_risk_and_correct_chain(self):
        by_class = {o["class_name"]: o for o in self.result["snapshot"]["objects"]}
        plate = by_class["plate"]
        self.assertGreater(plate["risk_score"], 0.0)
        risk_map = {o["track_id"]: o for o in self.result["snapshot"]["objects"]}
        chain = [risk_map[t]["class_name"] for t in plate["risk_chain"]]
        self.assertEqual(chain, ["nut_butter_jar", "cutlery", "bread", "plate"])
        self.assertEqual(plate["propagation_depth"], 2)

    def test_plate_never_directly_contacts_source(self):
        seen = {frozenset((c["source_class"], c["target_class"])) for c in self.result["contacts"]}
        self.assertNotIn(frozenset(("nut_butter_jar", "plate")), seen)

    def test_risk_decays_but_stays_positive_down_chain(self):
        by_class = {o["class_name"]: o for o in self.result["snapshot"]["objects"]}
        cutlery, bread, plate = by_class["cutlery"], by_class["bread"], by_class["plate"]
        for obj in (cutlery, bread, plate):
            self.assertGreater(obj["risk_score"], 0.0)
        # Monotonic non-increasing along the chain (tendency, not exact values).
        self.assertGreaterEqual(cutlery["risk_score"], bread["risk_score"])
        self.assertGreaterEqual(bread["risk_score"], plate["risk_score"])

    def test_overall_success_flag(self):
        self.assertTrue(self.result["checks"]["passed"])

    def test_physical_verification_is_false(self):
        self.assertFalse(self.result["physical_verification"])
        self.assertFalse(self.result["snapshot"]["physical_verification"])

    def test_snapshot_carries_dashboard_contract_fields(self):
        snap = self.result["snapshot"]
        self.assertEqual(snap["source"], "mock")
        self.assertEqual(snap["model"], "tracksense_8class_multiscene_best.pt")
        for key in ("objects", "timeline", "alerts", "explanations"):
            self.assertIn(key, snap)


class TestHeadlessArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run("flagship_chain", frames=150, model_path=_MODEL_PATH, out_dir=_OUT_DIR)

    def test_all_three_artifacts_written(self):
        for path in (self.result["events_path"], self.result["snapshot_path"],
                     self.result["summary_path"]):
            self.assertTrue(os.path.exists(path), f"missing artifact {path}")

    def test_events_jsonl_one_row_per_frame_and_parses(self):
        with open(self.result["events_path"], encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual(len(rows), self.result["frames_processed"])
        frames_with_contacts = [r["frame_index"] for r in rows if r["contacts"]]
        self.assertEqual(len(frames_with_contacts), 3)  # 3 frames each report 1 new contact

    def test_summary_contains_warning(self):
        with open(self.result["summary_path"], encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("Physical kitchen/camera validation has not yet been completed", text)


class TestHeadlessNonFlagship(unittest.TestCase):
    def test_safe_scenario_runs_and_stays_low(self):
        result = run("safe_unrelated_contacts", model_path=_MODEL_PATH, out_dir=_OUT_DIR)
        for obj in result["snapshot"]["objects"]:
            self.assertEqual(obj["risk_class"], "LOW")
        self.assertFalse(result["checks"].get("is_flagship"))


if __name__ == "__main__":
    unittest.main()
