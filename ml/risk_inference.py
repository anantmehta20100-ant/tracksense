"""Inference API for the Random Forest cross-contact risk model (Step 9).

Loads the trained pipeline ONCE (module-level cache), validates incoming
features against the centralized schema, and returns a structured prediction.

    from ml.risk_inference import predict_contact_risk
    predict_contact_risk({
        "source_object": "nut_butter_jar", "target_object": "bread",
        "source_current_risk": 1.0, "target_previous_risk": 0.0,
        "is_source_allergen": 1, "contact_duration": 6.0,
        "bbox_overlap_ratio": 0.4, "normalized_distance": 0.1,
        "time_since_last_contact": 0.0, "source_contact_count": 0,
        "target_contact_count": 0, "propagation_depth": 0,
        "cleaning_detected": 0, "repeated_contact_count": 0,
        "seconds_since_source_exposure": 0.0,
    })
    # -> {"risk_class": "HIGH", "risk_class_id": 2,
    #     "probabilities": {"LOW":.., "MEDIUM":.., "HIGH":..},
    #     "risk_score": .., "model_version": "risk-rf-1.0.0"}

The preprocessing is exactly the training pipeline (same joblib artifact), so
feature ordering / encoding can never drift.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.risk_features import (  # noqa: E402
    RISK_CLASS_LABELS,
    RISK_ID_TO_CLASS,
    events_to_frame,
    risk_score_from_proba,
    validate_event,
)
from ml.train_random_forest import METADATA_PATH, MODEL_PATH  # noqa: E402

_MODEL = None
_METADATA = None
_LOADED_FROM = None


def load_model(model_path: str = MODEL_PATH, metadata_path: str = METADATA_PATH, *, force_reload: bool = False):
    """Load (and cache) the trained pipeline + metadata. Subsequent calls reuse
    the cached objects -- the model is never re-read per frame/event."""
    global _MODEL, _METADATA, _LOADED_FROM
    if _MODEL is not None and not force_reload and _LOADED_FROM == model_path:
        return _MODEL, _METADATA
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained risk model at '{model_path}'. Run ml/train_random_forest.py first."
        )
    _MODEL = joblib.load(model_path)
    _METADATA = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            _METADATA = json.load(handle)
    _LOADED_FROM = model_path
    return _MODEL, _METADATA


def model_version() -> str:
    _, metadata = load_model()
    return metadata.get("model_version", "unknown")


def predict_contact_risk(event_features: dict, *, clamp: bool = False,
                         model_path: str = None, metadata_path: str = None) -> dict:
    """Predict relative cross-contact risk for one contact event.

    `event_features` must contain every feature in
    ml.risk_features.FEATURE_ORDER (missing/invalid fields raise
    FeatureValidationError). With clamp=True, out-of-range numeric values are
    clipped into range instead of raising -- useful for noisy live inputs.
    `model_path`/`metadata_path` override the default artifacts (used by tests);
    the loaded model is still cached by path, so repeated calls don't re-read it.
    """
    model, metadata = load_model(model_path or MODEL_PATH, metadata_path or METADATA_PATH)
    validated = validate_event(event_features, clamp=clamp)
    frame = events_to_frame([validated], validate=False)

    classifier = model.named_steps["random_forest"]
    proba_row = model.predict_proba(frame)[0]
    probabilities = {label: 0.0 for label in RISK_CLASS_LABELS}
    for class_id, prob in zip(classifier.classes_, proba_row):
        probabilities[RISK_ID_TO_CLASS[int(class_id)]] = float(prob)

    predicted_id = int(model.predict(frame)[0])
    return {
        "risk_class": RISK_ID_TO_CLASS[predicted_id],
        "risk_class_id": predicted_id,
        "probabilities": probabilities,
        "risk_score": round(risk_score_from_proba(probabilities), 4),
        "model_version": metadata.get("model_version", "unknown"),
    }


def predict_contact_risk_batch(events, *, clamp: bool = False,
                               model_path: str = None, metadata_path: str = None) -> list:
    """Vectorized prediction for many events (validated individually)."""
    model, metadata = load_model(model_path or MODEL_PATH, metadata_path or METADATA_PATH)
    validated = [validate_event(e, clamp=clamp) for e in events]
    if not validated:
        return []
    frame = events_to_frame(validated, validate=False)

    classifier = model.named_steps["random_forest"]
    proba = model.predict_proba(frame)
    predicted_ids = model.predict(frame).astype(int)
    version = metadata.get("model_version", "unknown")

    results = []
    for row_idx, predicted_id in enumerate(predicted_ids):
        probabilities = {label: 0.0 for label in RISK_CLASS_LABELS}
        for col, class_id in enumerate(classifier.classes_):
            probabilities[RISK_ID_TO_CLASS[int(class_id)]] = float(proba[row_idx, col])
        results.append({
            "risk_class": RISK_ID_TO_CLASS[int(predicted_id)],
            "risk_class_id": int(predicted_id),
            "probabilities": probabilities,
            "risk_score": round(risk_score_from_proba(probabilities), 4),
            "model_version": version,
        })
    return results


if __name__ == "__main__":
    example = {
        "source_object": "nut_butter_jar", "target_object": "bread",
        "source_current_risk": 1.0, "target_previous_risk": 0.0,
        "is_source_allergen": 1, "contact_duration": 6.0,
        "bbox_overlap_ratio": 0.4, "normalized_distance": 0.1,
        "time_since_last_contact": 0.0, "source_contact_count": 0,
        "target_contact_count": 0, "propagation_depth": 0,
        "cleaning_detected": 0, "repeated_contact_count": 0,
        "seconds_since_source_exposure": 0.0,
    }
    print(json.dumps(predict_contact_risk(example), indent=2))
