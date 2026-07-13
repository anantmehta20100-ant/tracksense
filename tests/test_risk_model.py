"""Tests for the Random Forest cross-contact risk pipeline (Step 11).

Run from the project root:  python -m unittest tests.test_risk_model

A small model is trained ONCE (setUpModule) into a temp dir, so the suite is
fast, self-contained, and never touches the real model/dataset artifacts.
Tendency tests assert directional behavior ("cleaning lowers risk", "direct
allergen scores above unrelated"), never brittle exact probabilities.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASS_TO_ID, get_allergen_type  # noqa: E402
from ml.class_schema import canonical_to_model, model_to_canonical  # noqa: E402
from ml.generate_risk_training_data import generate_dataset  # noqa: E402
from ml.risk_features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CLEANING_SUPPLY_LABEL,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    OBJECT_FEATURE_CATEGORIES,
    RISK_CLASS_LABELS,
    RISK_CLASS_TO_ID,
    FeatureValidationError,
    feature_row,
    risk_score_from_proba,
    validate_event,
)
from ml.risk_inference import predict_contact_risk, predict_contact_risk_batch  # noqa: E402
from ml.train_random_forest import (  # noqa: E402
    GROUP_COLUMN,
    LABEL_COLUMN,
    parse_args,
    split_by_scenario,
    train,
    validate_dataframe,
)
from pipeline.risk_engine import RiskEngine  # noqa: E402

_TMP_DIR = None
_CSV_PATH = None
_MODEL_PATH = None
_META_PATH = None


def setUpModule():
    global _TMP_DIR, _CSV_PATH, _MODEL_PATH, _META_PATH
    _TMP_DIR = tempfile.mkdtemp(prefix="tracksense_rf_test_")
    _CSV_PATH = os.path.join(_TMP_DIR, "risk_events.csv")
    _MODEL_PATH = os.path.join(_TMP_DIR, "rf.joblib")
    _META_PATH = os.path.join(_TMP_DIR, "rf.json")

    events_df, _ = generate_dataset(num_scenarios=400, seed=42)
    events_df.to_csv(_CSV_PATH, index=False)

    args = parse_args([
        "--no-search", "--n-estimators", "80", "--min-samples-leaf", "2",
        "--data", _CSV_PATH, "--model-out", _MODEL_PATH,
        "--metadata-out", _META_PATH, "--seed", "42",
    ])
    train(args)


def tearDownModule():
    if _TMP_DIR:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)


def make_event(**overrides):
    """A schema-valid event with sensible defaults; override any field."""
    event = {name: 0 for name in NUMERIC_FEATURES}
    event.update({
        "source_object": "cutlery", "target_object": "bread",
        "bbox_overlap_ratio": 0.4, "normalized_distance": 0.1,
    })
    event.update(overrides)
    return event


class TestFeatureSchema(unittest.TestCase):
    def test_feature_order_and_counts(self):
        self.assertEqual(len(FEATURE_ORDER), 15)
        self.assertEqual(FEATURE_ORDER[:2], CATEGORICAL_FEATURES)
        self.assertEqual(len(NUMERIC_FEATURES), 13)
        self.assertFalse(any(f.startswith("latent_") for f in FEATURE_ORDER))

    def test_missing_field_raises(self):
        event = make_event()
        del event["propagation_depth"]
        with self.assertRaises(FeatureValidationError):
            validate_event(event)

    def test_out_of_range_raises_and_clamps(self):
        with self.assertRaises(FeatureValidationError):
            validate_event(make_event(source_current_risk=5.0))
        clamped = validate_event(make_event(source_current_risk=5.0), clamp=True)
        self.assertEqual(clamped["source_current_risk"], 1.0)

    def test_binary_rejects_non_binary(self):
        with self.assertRaises(FeatureValidationError):
            validate_event(make_event(is_source_allergen=2))

    def test_feature_row_matches_order(self):
        row = feature_row(make_event())
        self.assertEqual(len(row), len(FEATURE_ORDER))
        self.assertEqual(row[0], "cutlery")  # source_object first

    def test_risk_score_from_proba(self):
        self.assertEqual(risk_score_from_proba({"LOW": 1.0, "MEDIUM": 0.0, "HIGH": 0.0}), 0.0)
        self.assertEqual(risk_score_from_proba({"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 1.0}), 1.0)
        self.assertAlmostEqual(risk_score_from_proba([0.2, 0.5, 0.3]), 0.55)

    def test_object_categories_cover_all_classes(self):
        for class_name in OBJECT_CLASS_TO_ID:  # all 9 canonical classes
            self.assertIn(class_name, OBJECT_FEATURE_CATEGORIES)
        self.assertIn("bread", OBJECT_FEATURE_CATEGORIES)
        self.assertIn(CLEANING_SUPPLY_LABEL, OBJECT_FEATURE_CATEGORIES)


class TestDeterministicGeneration(unittest.TestCase):
    def test_same_seed_identical(self):
        df_a, _ = generate_dataset(num_scenarios=150, seed=42)
        df_b, _ = generate_dataset(num_scenarios=150, seed=42)
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_different_seed_differs(self):
        df_a, _ = generate_dataset(num_scenarios=150, seed=42)
        df_c, _ = generate_dataset(num_scenarios=150, seed=7)
        self.assertFalse(df_a.equals(df_c))

    def test_schema_columns_and_labels(self):
        df, _ = generate_dataset(num_scenarios=150, seed=42)
        for feature in FEATURE_ORDER:
            self.assertIn(feature, df.columns)
        self.assertTrue(set(df[LABEL_COLUMN].unique()).issubset(set(RISK_CLASS_TO_ID.values())))
        latent_cols = [c for c in df.columns if c.startswith("latent_")]
        self.assertTrue(latent_cols)  # latents exist for transparency
        self.assertTrue(set(latent_cols).isdisjoint(FEATURE_ORDER))  # but never as features


class TestScenarioSplitLeakage(unittest.TestCase):
    def setUp(self):
        self.df, _ = generate_dataset(num_scenarios=300, seed=42)

    def test_zero_scenario_overlap(self):
        train_df, val_df, test_df, (train_ids, val_ids, test_ids) = split_by_scenario(self.df, seed=42)
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))
        # No scenario_id appears in more than one split's rows.
        self.assertTrue(set(train_df[GROUP_COLUMN]).isdisjoint(set(test_df[GROUP_COLUMN])))
        self.assertTrue(set(train_df[GROUP_COLUMN]).isdisjoint(set(val_df[GROUP_COLUMN])))
        self.assertEqual(train_ids | val_ids | test_ids, set(self.df[GROUP_COLUMN].unique()))

    def test_fractions_approximately_70_15_15(self):
        _, _, _, (train_ids, val_ids, test_ids) = split_by_scenario(self.df, seed=42)
        total = len(train_ids) + len(val_ids) + len(test_ids)
        self.assertAlmostEqual(len(train_ids) / total, 0.70, delta=0.03)
        self.assertAlmostEqual(len(val_ids) / total, 0.15, delta=0.03)
        self.assertAlmostEqual(len(test_ids) / total, 0.15, delta=0.03)


class TestTrainingArtifacts(unittest.TestCase):
    def test_model_and_metadata_saved(self):
        self.assertTrue(os.path.exists(_MODEL_PATH))
        self.assertTrue(os.path.exists(_META_PATH))

    def test_metadata_contents(self):
        import json
        with open(_META_PATH, encoding="utf-8") as handle:
            meta = json.load(handle)
        self.assertEqual(meta["feature_order"], FEATURE_ORDER)
        self.assertEqual(meta["class_labels"], RISK_CLASS_LABELS)
        self.assertEqual(meta["random_seed"], 42)
        self.assertIn("sklearn_version", meta)
        self.assertIn("hyperparameters", meta)
        self.assertIn("training_timestamp", meta)

    def test_validate_dataframe_rejects_missing_columns(self):
        df, _ = generate_dataset(num_scenarios=50, seed=42)
        with self.assertRaises(ValueError):
            validate_dataframe(df.drop(columns=["propagation_depth"]))


class TestInference(unittest.TestCase):
    def test_output_format(self):
        result = predict_contact_risk(
            make_event(source_object="nut_butter_jar", is_source_allergen=1,
                       source_current_risk=1.0, contact_duration=6.0),
            model_path=_MODEL_PATH, metadata_path=_META_PATH,
        )
        self.assertEqual(set(result), {"risk_class", "risk_class_id", "probabilities",
                                       "risk_score", "model_version"})
        self.assertIn(result["risk_class"], RISK_CLASS_LABELS)
        self.assertEqual(result["risk_class_id"], RISK_CLASS_TO_ID[result["risk_class"]])
        self.assertEqual(set(result["probabilities"]), set(RISK_CLASS_LABELS))
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertGreaterEqual(result["risk_score"], 0.0)
        self.assertLessEqual(result["risk_score"], 1.0)

    def test_serialization_reload_consistent(self):
        import joblib
        event = make_event(source_object="nut_butter_jar", is_source_allergen=1, source_current_risk=1.0)
        first = predict_contact_risk(event, model_path=_MODEL_PATH, metadata_path=_META_PATH)

        # Genuine reload from disk -> same class, ~same probabilities (RF's
        # parallel tree averaging can differ by ~1 ULP, so compare approximately).
        reloaded = joblib.load(_MODEL_PATH)
        self.assertIsNotNone(reloaded)
        second = predict_contact_risk(event, model_path=_MODEL_PATH, metadata_path=_META_PATH)
        self.assertEqual(first["risk_class_id"], second["risk_class_id"])
        for label in RISK_CLASS_LABELS:
            self.assertAlmostEqual(first["probabilities"][label], second["probabilities"][label], places=6)

    def test_batch_matches_single(self):
        events = [make_event(source_current_risk=r) for r in (0.0, 0.5, 1.0)]
        batch = predict_contact_risk_batch(events, model_path=_MODEL_PATH, metadata_path=_META_PATH)
        self.assertEqual(len(batch), 3)
        single = predict_contact_risk(events[1], model_path=_MODEL_PATH, metadata_path=_META_PATH)
        self.assertEqual(batch[1]["risk_class_id"], single["risk_class_id"])

    def test_missing_feature_raises(self):
        event = make_event()
        del event["cleaning_detected"]
        with self.assertRaises(FeatureValidationError):
            predict_contact_risk(event, model_path=_MODEL_PATH, metadata_path=_META_PATH)


class TestRiskTendencies(unittest.TestCase):
    def test_direct_allergen_scores_above_unrelated(self):
        direct = [
            make_event(source_object=src, target_object="bread", is_source_allergen=1,
                       source_current_risk=1.0, propagation_depth=0, contact_duration=d,
                       bbox_overlap_ratio=0.5, normalized_distance=0.1)
            for src in ("nut_butter_jar", "whole_nuts") for d in (3.0, 6.0, 10.0)
        ]
        unrelated = [
            make_event(source_object="plate", target_object="bowl", is_source_allergen=0,
                       source_current_risk=0.0, target_previous_risk=0.0, propagation_depth=0,
                       contact_duration=d, bbox_overlap_ratio=0.3, normalized_distance=0.5)
            for d in (0.5, 1.0, 2.0, 3.0, 6.0, 10.0)
        ]
        direct_scores = [r["risk_score"] for r in predict_contact_risk_batch(
            direct, model_path=_MODEL_PATH, metadata_path=_META_PATH)]
        unrelated_scores = [r["risk_score"] for r in predict_contact_risk_batch(
            unrelated, model_path=_MODEL_PATH, metadata_path=_META_PATH)]
        mean_direct = sum(direct_scores) / len(direct_scores)
        mean_unrelated = sum(unrelated_scores) / len(unrelated_scores)
        self.assertGreater(mean_direct, mean_unrelated)

    def test_cleaning_generally_lowers_risk(self):
        risks = [0.4, 0.55, 0.7, 0.85, 1.0]
        cleaning = [
            make_event(source_object=CLEANING_SUPPLY_LABEL, target_object="bread",
                       is_source_allergen=0, source_current_risk=0.0, target_previous_risk=r,
                       cleaning_detected=1, propagation_depth=1, contact_duration=5.0)
            for r in risks
        ]
        ongoing = [
            make_event(source_object="cutlery", target_object="bread", is_source_allergen=0,
                       source_current_risk=r, target_previous_risk=0.1, cleaning_detected=0,
                       propagation_depth=1, contact_duration=5.0, bbox_overlap_ratio=0.5,
                       normalized_distance=0.1)
            for r in risks
        ]
        cleaning_scores = [r["risk_score"] for r in predict_contact_risk_batch(
            cleaning, model_path=_MODEL_PATH, metadata_path=_META_PATH)]
        ongoing_scores = [r["risk_score"] for r in predict_contact_risk_batch(
            ongoing, model_path=_MODEL_PATH, metadata_path=_META_PATH)]
        mean_cleaning = sum(cleaning_scores) / len(cleaning_scores)
        mean_ongoing = sum(ongoing_scores) / len(ongoing_scores)
        self.assertLess(mean_cleaning, mean_ongoing)


class TestClassMappings(unittest.TestCase):
    def test_canonical_ids_unchanged(self):
        expected = {
            "nut_butter_jar": 0, "whole_nuts": 1, "hand": 2, "cutlery": 3,
            "chopping_board": 4, "plate": 5, "bowl": 6, "counter": 7, "bread": 8,
        }
        for name, idx in expected.items():
            self.assertEqual(OBJECT_CLASS_TO_ID[name], idx)

    def test_bread_model_local_remap_intact(self):
        # bread canonical 8 <-> model-local 7; counter (7) excluded from training.
        self.assertEqual(canonical_to_model(8), 7)
        self.assertEqual(model_to_canonical(7), 8)

    def test_allergen_sources(self):
        self.assertEqual(get_allergen_type("nut_butter_jar"), "nut")
        self.assertEqual(get_allergen_type("whole_nuts"), "nut")
        self.assertIsNone(get_allergen_type("bread"))


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RiskEngine(model_path=_MODEL_PATH)

    def _event(self, s_id, s_cls, t_id, t_cls, ts):
        return {"source_track_id": s_id, "source_class": s_cls, "target_track_id": t_id,
                "target_class": t_cls, "timestamp": ts,
                "allergen_type": get_allergen_type(s_cls)}

    def test_build_features_valid_and_flags_allergen(self):
        event = self._event(1, "nut_butter_jar", 2, "cutlery", 100.0)
        features = self.engine.build_features(event)
        validate_event(features)  # must not raise
        self.assertEqual(features["is_source_allergen"], 1)
        self.assertEqual(features["propagation_depth"], 0)

    def test_process_chain_updates_state(self):
        results = []
        results.append(self.engine.process_contact_event(self._event(1, "nut_butter_jar", 2, "cutlery", 100.0)))
        results.append(self.engine.process_contact_event(self._event(2, "cutlery", 3, "bread", 103.0)))
        for r in results:
            self.assertIn(r["risk_class"], RISK_CLASS_LABELS)

        cutlery = self.engine.get_risk(2)
        bread = self.engine.get_risk(3)
        self.assertIsNotNone(cutlery)
        self.assertEqual(bread["propagation_depth"], 1)  # one hop past the direct-contact cutlery
        self.assertGreaterEqual(cutlery["contact_count"], 1)

    def test_reset_clears_state(self):
        self.engine.process_contact_event(self._event(1, "nut_butter_jar", 2, "cutlery", 100.0))
        self.assertTrue(self.engine.risk_map())
        self.engine.reset()
        self.assertEqual(self.engine.risk_map(), {})


if __name__ == "__main__":
    unittest.main()
