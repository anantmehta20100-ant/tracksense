"""Centralized feature schema for the Random Forest cross-contact risk model.

This is the SINGLE SOURCE OF TRUTH for:
  - the feature list and its exact order (never hardcode the order elsewhere),
  - each feature's kind / units / valid range,
  - the categorical object vocabulary,
  - the risk-class labels (LOW / MEDIUM / HIGH) and their ids,
  - the continuous "convenience score" derived from class probabilities.

Training (ml/train_random_forest.py), the synthetic generator
(ml/generate_risk_training_data.py), evaluation
(evaluate/evaluate_random_forest.py), inference (ml/risk_inference.py) and the
runtime engine (pipeline/risk_engine.py) all import from here, so the feature
order and preprocessing can never silently drift between training and inference.

The model predicts *relative cross-contact risk*, not a measured allergen
concentration. See README.md "Scientific honesty / limitations".

Run `python ml/risk_features.py` to execute the self-test.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable whether this file is run as `python
# ml/risk_features.py` or imported as `ml.risk_features` (matches the pattern
# used by ml/class_schema.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASSES  # noqa: E402

# ---------------------------------------------------------------------------
# Risk classes (the model output)
# ---------------------------------------------------------------------------
RISK_CLASS_LABELS = ["LOW", "MEDIUM", "HIGH"]
RISK_CLASS_TO_ID = {label: idx for idx, label in enumerate(RISK_CLASS_LABELS)}
RISK_ID_TO_CLASS = {idx: label for label, idx in RISK_CLASS_TO_ID.items()}

# Weights used to collapse the 3 class probabilities into one continuous
# convenience score in [0, 1] (Step 3 of the build spec). This is a convenience
# ordering aid, NOT a calibrated probability of contamination.
RISK_SCORE_WEIGHTS = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}

# ---------------------------------------------------------------------------
# Categorical object vocabulary
# ---------------------------------------------------------------------------
# "cleaning_supply" mirrors model/synthetic_data.CLEANING_SUPPLY_LABEL: a
# cleaning tool can be the *source* of a (cleaning) contact event even though it
# is not one of the detected YOLO object classes. Kept as an explicit category
# so train/inference share one stable one-hot vocabulary.
CLEANING_SUPPLY_LABEL = "cleaning_supply"
OBJECT_FEATURE_CATEGORIES = list(OBJECT_CLASSES) + [CLEANING_SUPPLY_LABEL]

# ---------------------------------------------------------------------------
# Feature schema (ORDER MATTERS -- this is the canonical order everywhere)
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["source_object", "target_object"]

NUMERIC_FEATURES = [
    "source_current_risk",
    "target_previous_risk",
    "is_source_allergen",
    "contact_duration",
    "bbox_overlap_ratio",
    "normalized_distance",
    "time_since_last_contact",
    "source_contact_count",
    "target_contact_count",
    "propagation_depth",
    "cleaning_detected",
    "repeated_contact_count",
    "seconds_since_source_exposure",
]

FEATURE_ORDER = CATEGORICAL_FEATURES + NUMERIC_FEATURES

BINARY_FEATURES = {"is_source_allergen", "cleaning_detected"}

# kind: "categorical" | "float" | "int" | "binary"
# (low, high): inclusive valid range used by validate_event(). None = unbounded
# on that side. `unit` documents the physical meaning.
FEATURE_SPECS = {
    "source_object": {
        "kind": "categorical",
        "unit": "class name",
        "range": None,
        "desc": "Object class the contamination is transferring FROM.",
    },
    "target_object": {
        "kind": "categorical",
        "unit": "class name",
        "range": None,
        "desc": "Object class the contamination is transferring TO (risk predicted for this).",
    },
    "source_current_risk": {
        "kind": "float",
        "unit": "relative risk score [0,1]",
        "range": (0.0, 1.0),
        "desc": "Engine's current risk estimate for the source object.",
    },
    "target_previous_risk": {
        "kind": "float",
        "unit": "relative risk score [0,1]",
        "range": (0.0, 1.0),
        "desc": "Engine's risk estimate for the target BEFORE this contact.",
    },
    "is_source_allergen": {
        "kind": "binary",
        "unit": "0/1",
        "range": (0, 1),
        "desc": "1 if the source is a raw allergen source class (nut_butter_jar / whole_nuts).",
    },
    "contact_duration": {
        "kind": "float",
        "unit": "seconds",
        "range": (0.0, 600.0),
        "desc": "How long the two objects stayed in contact.",
    },
    "bbox_overlap_ratio": {
        "kind": "float",
        "unit": "IoU-like ratio [0,1]",
        "range": (0.0, 1.0),
        "desc": "Bounding-box overlap between source and target during contact.",
    },
    "normalized_distance": {
        "kind": "float",
        "unit": "distance / object size",
        "range": (0.0, 10.0),
        "desc": "Center gap normalized by object size (0 = fully overlapping).",
    },
    "time_since_last_contact": {
        "kind": "float",
        "unit": "seconds",
        "range": (0.0, 86400.0),
        "desc": "Seconds since the target's previous contact of any kind.",
    },
    "source_contact_count": {
        "kind": "int",
        "unit": "count",
        "range": (0, 100000),
        "desc": "How many contact events the source object has been involved in so far.",
    },
    "target_contact_count": {
        "kind": "int",
        "unit": "count",
        "range": (0, 100000),
        "desc": "How many contact events the target object has been involved in so far.",
    },
    "propagation_depth": {
        "kind": "int",
        "unit": "hops",
        "range": (0, 1000),
        "desc": "Number of hops from the original allergen source (0 = direct source contact).",
    },
    "cleaning_detected": {
        "kind": "binary",
        "unit": "0/1",
        "range": (0, 1),
        "desc": "1 if a cleaning action was detected on the target around this contact.",
    },
    "repeated_contact_count": {
        "kind": "int",
        "unit": "count",
        "range": (0, 100000),
        "desc": "How many times this exact source->target pair has contacted before.",
    },
    "seconds_since_source_exposure": {
        "kind": "float",
        "unit": "seconds",
        "range": (0.0, 86400.0),
        "desc": "Seconds since the original allergen source first entered this chain.",
    },
}

assert set(FEATURE_SPECS) == set(FEATURE_ORDER), "FEATURE_SPECS must cover exactly FEATURE_ORDER"
assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER)), "duplicate feature name"


class FeatureValidationError(ValueError):
    """Raised when an event dict does not satisfy the feature schema."""


def _coerce_number(name: str, value):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(f"feature '{name}' is not numeric: {value!r}") from exc


def validate_event(event: dict, *, clamp: bool = False) -> dict:
    """Validate one contact-event feature dict against the schema.

    Returns a NEW dict containing exactly FEATURE_ORDER keys, with numeric
    values coerced to float/int and categoricals to str. Raises
    FeatureValidationError on missing fields, wrong types, or out-of-range
    values. With clamp=True, numeric values are clipped into range instead of
    raising on range violations (categorical/missing errors still raise).

    Unknown object categories are allowed (the one-hot encoder is configured
    with handle_unknown="ignore"), so runtime never crashes on a class the
    training vocabulary did not contain.
    """
    if not isinstance(event, dict):
        raise FeatureValidationError(f"event must be a dict, got {type(event).__name__}")

    missing = [f for f in FEATURE_ORDER if f not in event]
    if missing:
        raise FeatureValidationError(f"missing required feature(s): {missing}")

    out = {}
    for name in CATEGORICAL_FEATURES:
        value = event[name]
        if not isinstance(value, str) or not value:
            raise FeatureValidationError(f"categorical feature '{name}' must be a non-empty str, got {value!r}")
        out[name] = value

    for name in NUMERIC_FEATURES:
        spec = FEATURE_SPECS[name]
        number = _coerce_number(name, event[name])
        low, high = spec["range"]

        if name in BINARY_FEATURES:
            if number not in (0.0, 1.0):
                raise FeatureValidationError(f"binary feature '{name}' must be 0 or 1, got {number}")

        if low is not None and number < low:
            if clamp:
                number = float(low)
            else:
                raise FeatureValidationError(f"feature '{name}'={number} below minimum {low}")
        if high is not None and number > high:
            if clamp:
                number = float(high)
            else:
                raise FeatureValidationError(f"feature '{name}'={number} above maximum {high}")

        out[name] = int(round(number)) if spec["kind"] in ("int", "binary") else number

    return out


def feature_row(event: dict, *, clamp: bool = False) -> list:
    """Return an event's feature values as a list in canonical FEATURE_ORDER."""
    validated = validate_event(event, clamp=clamp)
    return [validated[name] for name in FEATURE_ORDER]


def events_to_frame(events, *, validate: bool = True, clamp: bool = False):
    """Turn an iterable of event dicts into a DataFrame whose columns are
    exactly FEATURE_ORDER (in order). Import pandas lazily so this module stays
    importable in minimal environments."""
    import pandas as pd

    rows = []
    for event in events:
        rows.append(validate_event(event, clamp=clamp) if validate else {k: event[k] for k in FEATURE_ORDER})
    frame = pd.DataFrame(rows, columns=FEATURE_ORDER)
    for name in NUMERIC_FEATURES:
        frame[name] = frame[name].astype("int64" if FEATURE_SPECS[name]["kind"] in ("int", "binary") else "float64")
    return frame


def risk_score_from_proba(proba) -> float:
    """Collapse class probabilities into one continuous convenience score in
    [0, 1]: P(MEDIUM)*0.5 + P(HIGH)*1.0.

    Accepts a mapping {label: prob} or a sequence aligned with
    RISK_CLASS_LABELS.
    """
    if isinstance(proba, dict):
        values = {label: float(proba.get(label, 0.0)) for label in RISK_CLASS_LABELS}
    else:
        seq = list(proba)
        if len(seq) != len(RISK_CLASS_LABELS):
            raise ValueError(f"expected {len(RISK_CLASS_LABELS)} probabilities, got {len(seq)}")
        values = {label: float(seq[idx]) for idx, label in enumerate(RISK_CLASS_LABELS)}
    return sum(values[label] * RISK_SCORE_WEIGHTS[label] for label in RISK_CLASS_LABELS)


def _selftest() -> None:
    assert FEATURE_ORDER[:2] == ["source_object", "target_object"]
    assert len(NUMERIC_FEATURES) == 13
    assert len(FEATURE_ORDER) == 15
    assert RISK_CLASS_TO_ID == {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    # No latent/label column can be a feature (leakage guard used by tests).
    for name in FEATURE_ORDER:
        assert not name.startswith("latent_")
        assert name not in ("risk_class", "risk_class_id", "scenario_id")

    good = {name: 0 for name in NUMERIC_FEATURES}
    good.update({"source_object": "nut_butter_jar", "target_object": "bread",
                 "source_current_risk": 1.0, "is_source_allergen": 1,
                 "bbox_overlap_ratio": 0.3, "normalized_distance": 0.0})
    validated = validate_event(good)
    assert list(validated) == FEATURE_ORDER
    assert feature_row(good)[0] == "nut_butter_jar"

    # Missing field raises.
    bad = dict(good)
    del bad["propagation_depth"]
    try:
        validate_event(bad)
        raise AssertionError("expected missing-field error")
    except FeatureValidationError:
        pass

    # Out-of-range raises, or clamps.
    over = dict(good, source_current_risk=5.0)
    try:
        validate_event(over)
        raise AssertionError("expected range error")
    except FeatureValidationError:
        pass
    assert validate_event(over, clamp=True)["source_current_risk"] == 1.0

    # Binary must be 0/1.
    try:
        validate_event(dict(good, is_source_allergen=2))
        raise AssertionError("expected binary error")
    except FeatureValidationError:
        pass

    # Convenience score.
    assert risk_score_from_proba({"LOW": 1.0, "MEDIUM": 0.0, "HIGH": 0.0}) == 0.0
    assert risk_score_from_proba({"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 1.0}) == 1.0
    assert abs(risk_score_from_proba([0.2, 0.5, 0.3]) - (0.5 * 0.5 + 0.3 * 1.0)) < 1e-9

    frame = events_to_frame([good, good])
    assert list(frame.columns) == FEATURE_ORDER
    assert len(frame) == 2

    print("risk_features self-test PASS")
    print(f"  {len(FEATURE_ORDER)} features:", FEATURE_ORDER)
    print("  classes:", RISK_CLASS_LABELS)
    print("  object categories:", OBJECT_FEATURE_CATEGORIES)


if __name__ == "__main__":
    _selftest()
