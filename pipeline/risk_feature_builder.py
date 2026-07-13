"""Bridge from a runtime contact + engine state to the RF feature schema (Phase 6).

`build_risk_features` is the ONE place that turns "a contact happened between
these two objects, here is what we know about each" into the exact feature dict
the Random Forest expects. It imports the canonical order/spec from
ml/risk_features.py and never re-declares it, so training and inference can't
drift (the returned dict is validated against that schema before inference).

`source_state` / `target_state` are the engine's per-object states (or None if
the object is new). They only need to expose these attributes, so both the
engine's internal state object and pipeline.contracts.ObjectRiskState work:

    .risk_score, .propagation_depth, .source_exposure_time,
    .last_contact_time, .contact_count

Any runtime signal the vision layer cannot yet measure (duration, overlap,
distance, cleaning) comes in through `observations` with documented defaults --
values are never silently invented here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import get_allergen_type  # noqa: E402
from ml.risk_features import FEATURE_ORDER  # noqa: E402

# Documented placeholder observations for feature fields the current contact
# heuristic may not measure. The live ContactTracker DOES measure duration /
# overlap / distance, so these are only a fallback (e.g. a manually injected
# cleaning event) -- never a fabricated contact.
DEFAULT_OBSERVATIONS = {
    "contact_duration": 3.0,      # seconds; a sustained detected contact
    "bbox_overlap_ratio": 0.4,    # IoU-like overlap
    "normalized_distance": 0.1,   # near-touching
    "cleaning_detected": 0,       # no live cleaning detector
}


def _risk(state):
    return getattr(state, "risk_score", 0.0) if state is not None else 0.0


def build_risk_features(contact_event: dict, source_state, target_state,
                        repeated_contact_count: int, observations: dict = None) -> dict:
    """Assemble the full FEATURE_ORDER feature dict for one directed contact.

    contact_event: {source_track_id, source_class, target_track_id,
                    target_class, timestamp, ...}. Direction is already decided
    (source == where contamination flows from).
    """
    obs = dict(DEFAULT_OBSERVATIONS)
    if observations:
        obs.update(observations)

    source_class = contact_event["source_class"]
    target_class = contact_event["target_class"]
    timestamp = contact_event["timestamp"]

    is_source_allergen = 1 if get_allergen_type(source_class) is not None else 0

    if is_source_allergen:
        source_current_risk = 1.0            # a raw allergen source is fully contaminating
        propagation_depth = 0
        seconds_since_source_exposure = 0.0
    else:
        source_current_risk = _risk(source_state)
        if source_state is not None:
            propagation_depth = getattr(source_state, "propagation_depth", 0) + 1
            exposure = getattr(source_state, "source_exposure_time", None)
            seconds_since_source_exposure = max(0.0, timestamp - exposure) if exposure is not None else 0.0
        else:
            propagation_depth = 1
            seconds_since_source_exposure = 0.0

    target_previous_risk = _risk(target_state)
    last_contact = getattr(target_state, "last_contact_time", None) if target_state is not None else None
    time_since_last_contact = max(0.0, timestamp - last_contact) if last_contact is not None else 0.0

    features = {
        "source_object": source_class,
        "target_object": target_class,
        "source_current_risk": source_current_risk,
        "target_previous_risk": target_previous_risk,
        "is_source_allergen": is_source_allergen,
        "contact_duration": obs["contact_duration"],
        "bbox_overlap_ratio": obs["bbox_overlap_ratio"],
        "normalized_distance": obs["normalized_distance"],
        "time_since_last_contact": time_since_last_contact,
        "source_contact_count": getattr(source_state, "contact_count", 0) if source_state is not None else 0,
        "target_contact_count": getattr(target_state, "contact_count", 0) if target_state is not None else 0,
        "propagation_depth": propagation_depth,
        "cleaning_detected": int(obs["cleaning_detected"]),
        "repeated_contact_count": int(repeated_contact_count),
        "seconds_since_source_exposure": seconds_since_source_exposure,
    }
    # Defensive: guarantee we produced exactly the schema's keys.
    assert set(features) == set(FEATURE_ORDER), "feature builder drifted from FEATURE_ORDER"
    return features


def build_cleaning_features(target_state, timestamp: float, *,
                            cleaning_supply_label: str, observations: dict = None) -> dict:
    """Features for a cleaning action on `target_state` (Phase 10).

    Mirrors the synthetic generator's cleaning event: a cleaning-supply source
    with zero risk, cleaning_detected=1, the object's pre-clean risk as
    target_previous_risk, and its current depth preserved. The RF then predicts
    the (lower) residual risk -- no hard reset is fabricated.
    """
    obs = {"contact_duration": 5.0, "bbox_overlap_ratio": 0.4, "normalized_distance": 0.1}
    if observations:
        obs.update(observations)
    last_contact = getattr(target_state, "last_contact_time", None)
    time_since_last_contact = max(0.0, timestamp - last_contact) if last_contact is not None else 0.0
    exposure = getattr(target_state, "source_exposure_time", None)
    seconds_since_source_exposure = max(0.0, timestamp - exposure) if exposure is not None else 0.0

    features = {
        "source_object": cleaning_supply_label,
        "target_object": getattr(target_state, "class_name", "unknown"),
        "source_current_risk": 0.0,
        "target_previous_risk": _risk(target_state),
        "is_source_allergen": 0,
        "contact_duration": obs["contact_duration"],
        "bbox_overlap_ratio": obs["bbox_overlap_ratio"],
        "normalized_distance": obs["normalized_distance"],
        "time_since_last_contact": time_since_last_contact,
        "source_contact_count": 0,
        "target_contact_count": getattr(target_state, "contact_count", 0),
        "propagation_depth": getattr(target_state, "propagation_depth", 0),
        "cleaning_detected": 1,
        "repeated_contact_count": 0,
        "seconds_since_source_exposure": seconds_since_source_exposure,
    }
    assert set(features) == set(FEATURE_ORDER), "cleaning feature builder drifted from FEATURE_ORDER"
    return features
