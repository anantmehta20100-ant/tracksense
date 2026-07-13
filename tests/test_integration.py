"""Integration tests for the live detection->contact->RF->engine->API flow
(Phases 17 & 18).

A small RF is trained ONCE into a temp dir (setUpModule), so the suite is fast
and never touches the real 49MB model. The single most important test is the
flagship end-to-end propagation chain (TestFlagshipEndToEnd).

Run:  python -m unittest discover -s tests -p "test_integration.py"
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import get_allergen_type  # noqa: E402
from config.runtime_config import EXPECTED_YOLO_CLASS_NAMES  # noqa: E402
from ml.class_schema import canonical_to_model, model_to_canonical  # noqa: E402
from ml.generate_risk_training_data import generate_dataset  # noqa: E402
from ml.risk_features import FEATURE_ORDER, FeatureValidationError, validate_event  # noqa: E402
import ml.risk_inference as risk_inference  # noqa: E402
from ml.risk_inference import predict_contact_risk  # noqa: E402
from ml.train_random_forest import parse_args, train  # noqa: E402
from pipeline.contracts import ContactEvent, Detection  # noqa: E402
from pipeline.demo_controller import DemoController  # noqa: E402
from pipeline.risk_pipeline import RiskPipeline  # noqa: E402
from vision.contact_tracker import ContactTracker  # noqa: E402
from vision.mock_detection_source import MockDetectionSource, list_scenarios  # noqa: E402
from vision.tracker import IoUTracker, Track  # noqa: E402
from vision.yolo_detection_source import ModelSchemaMismatch, validate_class_names  # noqa: E402

_TMP_DIR = None
_MODEL_PATH = None
_META_PATH = None


def setUpModule():
    global _TMP_DIR, _MODEL_PATH, _META_PATH
    _TMP_DIR = tempfile.mkdtemp(prefix="tracksense_integ_")
    csv_path = os.path.join(_TMP_DIR, "risk_events.csv")
    _MODEL_PATH = os.path.join(_TMP_DIR, "rf.joblib")
    _META_PATH = os.path.join(_TMP_DIR, "rf.json")
    events_df, _ = generate_dataset(num_scenarios=400, seed=42)
    events_df.to_csv(csv_path, index=False)
    args = parse_args(["--no-search", "--n-estimators", "80", "--min-samples-leaf", "2",
                       "--data", csv_path, "--model-out", _MODEL_PATH,
                       "--metadata-out", _META_PATH, "--seed", "42"])
    train(args)


def tearDownModule():
    if _TMP_DIR:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)


def _track(track_id, class_name, bbox):
    return Track(track_id, {"class_name": class_name, "confidence": 0.9, "bbox": list(bbox)})


def _run_scenario(scenario):
    ctl = DemoController(scenario, model_path=_MODEL_PATH)
    return ctl.run_to_completion()


class TestMockSourceDeterminism(unittest.TestCase):
    def test_same_seed_identical_detections(self):
        a = [[d.to_dict() for d in fd.detections] for fd in MockDetectionSource("flagship_chain", seed=42).frames()]
        b = [[d.to_dict() for d in fd.detections] for fd in MockDetectionSource("flagship_chain", seed=42).frames()]
        self.assertEqual(a, b)

    def test_all_scenarios_present(self):
        self.assertEqual(set(list_scenarios()),
                         {"flagship_chain", "direct_source_contact", "cleaning_interrupts_chain",
                          "safe_unrelated_contacts", "repeated_contact"})


class TestTrackingStability(unittest.TestCase):
    def test_stable_ids_every_scenario(self):
        for scenario in list_scenarios():
            tracker = IoUTracker()
            seen = {}
            for fd in MockDetectionSource(scenario).frames():
                for t in tracker.update([d.to_tracker_dict() for d in fd.detections]):
                    seen.setdefault(t.class_name, set()).add(t.track_id)
            for cls, ids in seen.items():
                self.assertEqual(len(ids), 1, f"{scenario}: {cls} got ids {ids}")


class TestContactLifecycle(unittest.TestCase):
    def test_persistence_then_single_event_on_end(self):
        ct = ContactTracker()
        close_a = _track(1, "nut_butter_jar", (100, 100, 180, 220))
        close_b = _track(2, "cutlery", (150, 100, 230, 220))   # overlaps -> close
        far_b = _track(2, "cutlery", (400, 100, 480, 220))     # separated -> far

        persistence = ct.cfg.start_persistence_frames
        # Fewer than `persistence` close frames: still PENDING, no ended event.
        for f in range(persistence - 1):
            self.assertEqual(ct.update([close_a, close_b], f, f * 0.1), [])
        self.assertEqual(ct.active_contacts()[0]["phase"], "pending")

        # The persistence-th close frame confirms the contact (ACTIVE), still no event.
        self.assertEqual(ct.update([close_a, close_b], persistence - 1, (persistence - 1) * 0.1), [])
        self.assertEqual(ct.active_contacts()[0]["phase"], "active")

        # More close frames: no per-frame emission.
        for f in range(persistence, persistence + 4):
            self.assertEqual(ct.update([close_a, close_b], f, f * 0.1), [])

        # Separate: only after end_persistence far frames does exactly ONE event fire.
        base = persistence + 4
        emitted = []
        for i in range(ct.cfg.end_persistence_frames):
            emitted += ct.update([close_a, far_b], base + i, (base + i) * 0.1)
        self.assertEqual(len(emitted), 1)
        event = emitted[0]
        self.assertIsInstance(event, ContactEvent)
        self.assertGreater(event.duration, 0.0)
        self.assertGreaterEqual(event.overlap_ratio, 0.0)
        # Not re-emitted afterwards.
        self.assertEqual(ct.update([close_a, far_b], base + 9, (base + 9) * 0.1), [])


class TestFeatureSchemaBridge(unittest.TestCase):
    def test_engine_features_match_rf_schema(self):
        pipe = RiskPipeline(model_path=_MODEL_PATH)
        event = {"source_track_id": 1, "source_class": "nut_butter_jar", "target_track_id": 2,
                 "target_class": "cutlery", "timestamp": 1.0, "allergen_type": "nut"}
        features = pipe.engine.build_features(event)
        self.assertEqual(set(features), set(FEATURE_ORDER))
        validate_event(features)  # must not raise

    def test_missing_feature_raises_useful_error(self):
        bad = {name: 0 for name in FEATURE_ORDER}
        bad.update({"source_object": "cutlery", "target_object": "bread"})
        del bad["propagation_depth"]
        with self.assertRaises(FeatureValidationError):
            predict_contact_risk(bad, model_path=_MODEL_PATH, metadata_path=_META_PATH)


class TestModelLoadsOnce(unittest.TestCase):
    def test_inference_cache_is_reused(self):
        predict_contact_risk(
            {name: 0 for name in FEATURE_ORDER} | {"source_object": "nut_butter_jar",
             "target_object": "cutlery", "source_current_risk": 1.0, "is_source_allergen": 1},
            model_path=_MODEL_PATH, metadata_path=_META_PATH)
        first = risk_inference._MODEL
        predict_contact_risk(
            {name: 0 for name in FEATURE_ORDER} | {"source_object": "cutlery",
             "target_object": "bread"},
            model_path=_MODEL_PATH, metadata_path=_META_PATH)
        self.assertIs(risk_inference._MODEL, first)  # same cached object, not reloaded


class TestSourceInitialization(unittest.TestCase):
    def test_source_marked_and_non_source_low(self):
        pipe = RiskPipeline(model_path=_MODEL_PATH)
        pipe.engine.process_contact_event(
            {"source_track_id": 5, "source_class": "nut_butter_jar", "target_track_id": 6,
             "target_class": "cutlery", "timestamp": 1.0, "allergen_type": "nut"})
        source = pipe.engine.get_risk(5)
        self.assertTrue(source["is_allergen_source"])
        self.assertEqual(source["risk_class"], "HIGH")
        self.assertEqual(source["risk_chain"], [5])
        self.assertEqual(source["root_allergen_track_id"], 5)

    def test_unrelated_objects_stay_low(self):
        snap = _run_scenario("safe_unrelated_contacts")
        for obj in snap["objects"]:
            self.assertEqual(obj["risk_class"], "LOW")
            self.assertFalse(obj["is_allergen_source"])


class TestCleaning(unittest.TestCase):
    def test_cleaning_lowers_object_risk(self):
        pipe = RiskPipeline(model_path=_MODEL_PATH)
        pipe.engine.process_contact_event(
            {"source_track_id": 1, "source_class": "nut_butter_jar", "target_track_id": 2,
             "target_class": "cutlery", "timestamp": 1.0, "allergen_type": "nut"})
        before = pipe.engine.get_risk(2)["risk_score"]
        pipe.engine.mark_cleaned(2, timestamp=2.0)
        after = pipe.engine.get_risk(2)["risk_score"]
        self.assertLess(after, before)

    def test_cleaning_scenario_interrupts_chain(self):
        cleaned = _run_scenario("cleaning_interrupts_chain")
        flagship = _run_scenario("flagship_chain")
        bread_cleaned = next(o for o in cleaned["objects"] if o["class_name"] == "bread")
        bread_flagship = next(o for o in flagship["objects"] if o["class_name"] == "bread")
        # Bread downstream of a CLEANED cutlery ends lower than in the un-cleaned chain.
        self.assertLess(bread_cleaned["risk_score"], bread_flagship["risk_score"])


class TestReset(unittest.TestCase):
    def test_reset_clears_state(self):
        pipe = RiskPipeline(model_path=_MODEL_PATH)
        pipe.engine.process_contact_event(
            {"source_track_id": 1, "source_class": "nut_butter_jar", "target_track_id": 2,
             "target_class": "cutlery", "timestamp": 1.0, "allergen_type": "nut"})
        self.assertTrue(pipe.engine.risk_map())
        pipe.reset()
        self.assertEqual(pipe.engine.risk_map(), {})


class TestYoloAdapterSchema(unittest.TestCase):
    def test_bread_is_local_id_7(self):
        self.assertEqual(EXPECTED_YOLO_CLASS_NAMES[7], "bread")
        self.assertEqual(canonical_to_model(8), 7)
        self.assertEqual(model_to_canonical(7), 8)

    def test_counter_absent_from_detector_schema(self):
        self.assertNotIn("counter", EXPECTED_YOLO_CLASS_NAMES.values())
        with self.assertRaises(KeyError):
            canonical_to_model(7)  # counter has no model-local id

    def test_rejects_mismatched_class_names(self):
        with self.assertRaises(ModelSchemaMismatch):
            validate_class_names({0: "cutlery"})            # old single-class model
        self.assertEqual(validate_class_names(EXPECTED_YOLO_CLASS_NAMES),
                         {int(k): v for k, v in EXPECTED_YOLO_CLASS_NAMES.items()})


class TestFlagshipEndToEnd(unittest.TestCase):
    """The single most important test: source -> cutlery -> bread -> plate."""

    @classmethod
    def setUpClass(cls):
        cls.snap = _run_scenario("flagship_chain")
        cls.by_class = {o["class_name"]: o for o in cls.snap["objects"]}

    def test_all_expected_objects_present(self):
        for name in ("nut_butter_jar", "cutlery", "bread", "plate"):
            self.assertIn(name, self.by_class)

    def test_source_is_root_allergen(self):
        jar = self.by_class["nut_butter_jar"]
        self.assertTrue(jar["is_allergen_source"])
        for name in ("cutlery", "bread", "plate"):
            self.assertEqual(self.by_class[name]["root_allergen_track_id"], jar["track_id"])

    def test_downstream_objects_receive_risk(self):
        # cutlery (direct) elevated; risk decays along the chain but stays > 0.
        self.assertNotEqual(self.by_class["cutlery"]["risk_class"], "LOW")
        self.assertGreater(self.by_class["cutlery"]["risk_score"], 0.0)
        self.assertGreater(self.by_class["bread"]["risk_score"], 0.0)
        self.assertGreater(self.by_class["plate"]["risk_score"], 0.0)
        self.assertGreaterEqual(self.by_class["cutlery"]["risk_score"],
                                self.by_class["plate"]["risk_score"])

    def test_plate_chain_provenance(self):
        plate = self.by_class["plate"]
        risk_map = {o["track_id"]: o for o in self.snap["objects"]}
        chain_classes = [risk_map[tid]["class_name"] for tid in plate["risk_chain"]]
        self.assertEqual(chain_classes, ["nut_butter_jar", "cutlery", "bread", "plate"])
        self.assertIsNotNone(get_allergen_type(chain_classes[0]))  # starts at an allergen source
        self.assertEqual(chain_classes[-1], "plate")               # ends at plate
        self.assertEqual(plate["propagation_depth"], 2)

    def test_no_direct_source_to_plate_contact(self):
        contacts = [e for e in self.snap["timeline"] if e["type"] == "contact"]
        jar_id = self.by_class["nut_butter_jar"]["track_id"]
        plate_id = self.by_class["plate"]["track_id"]
        for e in contacts:
            pair = {e["source_track_id"], e["target_track_id"]}
            self.assertNotEqual(pair, {jar_id, plate_id},
                                "plate must never directly contact the allergen source")

    def test_explanation_renders_chain_text(self):
        # bread is a downstream MEDIUM+ object in both the tiny test model and the
        # full model, so its rendered chain_text is a stable check of the explainer.
        # (plate's full provenance chain is asserted in test_plate_chain_provenance;
        # its exact risk CLASS is model-dependent, so it is not asserted here.)
        exps = {e["object"]: e for e in self.snap["explanations"]}
        self.assertIn("bread", exps)
        self.assertEqual(exps["bread"]["chain_text"], "nut_butter_jar → cutlery → bread")
        for exp in self.snap["explanations"]:
            self.assertTrue(exp["chain_text"].startswith("nut_butter_jar"))


if __name__ == "__main__":
    unittest.main()
