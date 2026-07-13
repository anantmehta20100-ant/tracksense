"""Runtime risk engine: turns live contact events into per-object risk state
using the Random Forest model (Step 10 + integration Phases 8-10).

    detections -> tracker -> contact_tracker -> [ContactEvent]
        -> RiskEngine.process_contact_event()  (builds features, calls the RF)
        -> per-object risk state + PROPAGATION CHAIN provenance -> dashboard

It is a SEPARATE, parallel consumer to pipeline/risk_state.py (the GRU engine);
it does not modify or replace it. Both accept the same contact_event dict shape
emitted by vision/contact_detector.py and vision/contact_tracker.py:

    {source_track_id, source_class, target_track_id, target_class,
     timestamp, allergen_type}

What this engine maintains per tracked object (keyed by track_id):
  * relative risk score / class / probabilities (from the RF, never copied),
  * propagation_depth (hops from the original allergen source),
  * the full risk_chain of track_ids from the root allergen down to this object,
    plus parent_track_id and root_allergen_track_id (provenance for the "why"),
  * source-exposure clock and contact counts.

Feature construction is delegated to pipeline/risk_feature_builder.py (one
place, one schema). The RF predicts RELATIVE cross-contact risk, not measured
allergen concentration (see README limitations).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.risk_features import CLEANING_SUPPLY_LABEL, validate_event  # noqa: E402
from ml.risk_inference import load_model, predict_contact_risk  # noqa: E402
from pipeline.risk_feature_builder import (  # noqa: E402
    DEFAULT_OBSERVATIONS,
    build_cleaning_features,
    build_risk_features,
)


class _ObjectRisk:
    """Live risk state for one tracked object (keyed by track_id)."""

    __slots__ = ("track_id", "class_name", "risk_score", "risk_class", "risk_class_id",
                 "contact_count", "last_contact_time", "propagation_depth",
                 "source_exposure_time", "last_probabilities", "parent_track_id",
                 "root_allergen_track_id", "risk_chain", "last_updated",
                 "is_allergen_source", "last_cleaned_time")

    def __init__(self, track_id, class_name):
        self.track_id = track_id
        self.class_name = class_name
        self.risk_score = 0.0
        self.risk_class = "LOW"
        self.risk_class_id = 0
        self.contact_count = 0
        self.last_contact_time = None
        self.propagation_depth = 0
        self.source_exposure_time = None
        self.last_probabilities = {"LOW": 1.0, "MEDIUM": 0.0, "HIGH": 0.0}
        # provenance
        self.parent_track_id = None
        self.root_allergen_track_id = None
        self.risk_chain = []
        self.last_updated = None
        self.is_allergen_source = False
        self.last_cleaned_time = None

    def as_dict(self):
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "risk_score": round(self.risk_score, 4),
            "risk_class": self.risk_class,
            "risk_class_id": self.risk_class_id,
            "contact_count": self.contact_count,
            "propagation_depth": self.propagation_depth,
            "probabilities": dict(self.last_probabilities),
            # provenance (Phase 8/11)
            "parent_track_id": self.parent_track_id,
            "root_allergen_track_id": self.root_allergen_track_id,
            "risk_chain": list(self.risk_chain),
            "is_allergen_source": self.is_allergen_source,
            "last_updated": self.last_updated,
            "last_cleaned_time": self.last_cleaned_time,
        }


class RiskEngine:
    """Maintains per-object risk state + propagation provenance over time."""

    def __init__(self, model_path=None, *, load_eagerly: bool = True):
        self._model_path = model_path
        self._objects = {}                # track_id -> _ObjectRisk
        self._pair_repeats = {}           # (source_track_id, target_track_id) -> count
        if load_eagerly:
            load_model(**({"model_path": model_path} if model_path else {}))

    # -- state accessors ---------------------------------------------------
    def _object(self, track_id, class_name):
        obj = self._objects.get(track_id)
        if obj is None:
            obj = _ObjectRisk(track_id, class_name)
            self._objects[track_id] = obj
        else:
            obj.class_name = class_name
        return obj

    def get_risk(self, track_id):
        obj = self._objects.get(track_id)
        return obj.as_dict() if obj else None

    def risk_map(self):
        return {track_id: obj.as_dict() for track_id, obj in self._objects.items()}

    def reset(self):
        self._objects.clear()
        self._pair_repeats.clear()

    # -- feature construction ---------------------------------------------
    def build_features(self, contact_event: dict, observations: dict = None) -> dict:
        """Turn a contact_event (+ optional measured observations) into the RF
        feature dict. Pure/side-effect-free; delegates to the shared builder so
        the feature order lives in exactly one place."""
        source = self._objects.get(contact_event["source_track_id"])
        target = self._objects.get(contact_event["target_track_id"])
        pair = (contact_event["source_track_id"], contact_event["target_track_id"])
        repeated = self._pair_repeats.get(pair, 0)
        return build_risk_features(contact_event, source, target, repeated, observations)

    # -- provenance --------------------------------------------------------
    def _init_allergen_source(self, source: _ObjectRisk, timestamp):
        """A detected raw allergen source is the root of its own chain and, by
        definition, the highest-risk origin (Phase 9). This is not an RF
        prediction -- the source IS the allergen."""
        source.is_allergen_source = True
        source.root_allergen_track_id = source.track_id
        source.parent_track_id = None
        if not source.risk_chain:
            source.risk_chain = [source.track_id]
        if source.source_exposure_time is None:
            source.source_exposure_time = timestamp
        source.risk_score = 1.0
        source.risk_class = "HIGH"
        source.risk_class_id = 2
        source.last_probabilities = {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 1.0}
        source.last_updated = timestamp

    def _update_provenance(self, source: _ObjectRisk, target: _ObjectRisk):
        """Extend the source's chain to the target, acyclically. Only overwrites
        the target's chain when the source is itself rooted in an allergen (so a
        clean, unrelated contact never clobbers real provenance)."""
        src_chain = list(source.risk_chain) if source.risk_chain else (
            [source.track_id] if source.is_allergen_source else [])
        src_root = source.root_allergen_track_id

        if src_chain:
            if target.track_id in src_chain:
                proposed = src_chain[:src_chain.index(target.track_id) + 1]
            else:
                proposed = src_chain + [target.track_id]
        else:
            proposed = [source.track_id, target.track_id]

        if src_root is not None or target.root_allergen_track_id is None:
            target.risk_chain = proposed
            target.parent_track_id = source.track_id
            target.root_allergen_track_id = src_root

    # -- main entry point --------------------------------------------------
    def process_contact_event(self, contact_event: dict, observations: dict = None) -> dict:
        """Score one directed contact event and update the target's risk state
        and provenance. Returns the RF prediction dict augmented with the
        source/target track ids, or None if the event is malformed."""
        source_id = contact_event["source_track_id"]
        target_id = contact_event["target_track_id"]
        source_class = contact_event["source_class"]
        target_class = contact_event["target_class"]
        timestamp = contact_event["timestamp"]

        features = self.build_features(contact_event, observations)
        validate_event(features, clamp=True)   # clip noisy live inputs, don't crash
        prediction = predict_contact_risk(features, clamp=True, model_path=self._model_path)

        source = self._object(source_id, source_class)
        target = self._object(target_id, target_class)

        source.contact_count += 1
        target.contact_count += 1
        self._pair_repeats[(source_id, target_id)] = self._pair_repeats.get((source_id, target_id), 0) + 1

        if features["is_source_allergen"]:
            self._init_allergen_source(source, timestamp)

        # Update the target's risk state from the RF prediction (never copied).
        target.risk_score = prediction["risk_score"]
        target.risk_class = prediction["risk_class"]
        target.risk_class_id = prediction["risk_class_id"]
        target.last_probabilities = prediction["probabilities"]
        target.last_contact_time = timestamp
        target.propagation_depth = features["propagation_depth"]
        target.last_updated = timestamp

        self._update_provenance(source, target)

        # Propagate the "when did the allergen enter this chain" clock.
        if features["is_source_allergen"]:
            target.source_exposure_time = timestamp
        elif source.source_exposure_time is not None:
            if target.source_exposure_time is None:
                target.source_exposure_time = source.source_exposure_time
            else:
                target.source_exposure_time = min(target.source_exposure_time, source.source_exposure_time)

        return {**prediction, "target_track_id": target_id, "source_track_id": source_id}

    # -- cleaning ----------------------------------------------------------
    def mark_cleaned(self, track_id, timestamp: float = None, observations: dict = None) -> dict:
        """Apply a cleaning action to a tracked object (Phase 10).

        Runs the RF on a cleaning event (cleaning-supply source, cleaning_detected=1,
        the object's pre-clean risk as target_previous_risk) and lowers the
        object's risk to the RF's predicted residual. Risk is NOT hard-reset to
        zero -- the model decides how much cleaning helps, honestly reflecting
        that cleaning can be incomplete. Returns the prediction dict, or None if
        the object is unknown.
        """
        target = self._objects.get(track_id)
        if target is None:
            return None
        ts = timestamp if timestamp is not None else (target.last_updated or target.last_contact_time or 0.0)

        features = build_cleaning_features(
            target, ts, cleaning_supply_label=CLEANING_SUPPLY_LABEL, observations=observations)
        validate_event(features, clamp=True)
        prediction = predict_contact_risk(features, clamp=True, model_path=self._model_path)

        target.risk_score = prediction["risk_score"]
        target.risk_class = prediction["risk_class"]
        target.risk_class_id = prediction["risk_class_id"]
        target.last_probabilities = prediction["probabilities"]
        target.last_cleaned_time = ts
        target.last_updated = ts
        target.contact_count += 1
        return {**prediction, "target_track_id": track_id, "cleaned": True}


if __name__ == "__main__":
    import time

    if not os.path.exists("model/risk_random_forest.joblib"):
        raise SystemExit("Train the model first: python ml/train_random_forest.py")

    engine = RiskEngine()
    now = time.time()
    chain = [
        {"source_track_id": 1, "source_class": "nut_butter_jar", "target_track_id": 2,
         "target_class": "cutlery", "timestamp": now, "allergen_type": "nut"},
        {"source_track_id": 2, "source_class": "cutlery", "target_track_id": 3,
         "target_class": "bread", "timestamp": now + 3, "allergen_type": "nut"},
        {"source_track_id": 3, "source_class": "bread", "target_track_id": 4,
         "target_class": "plate", "timestamp": now + 6, "allergen_type": "nut"},
    ]
    for event in chain:
        result = engine.process_contact_event(event)
        print(f"{event['source_class']} -> {event['target_class']}: "
              f"{result['risk_class']} (score {result['risk_score']})")
    plate = engine.get_risk(4)
    print("plate chain (track ids):", plate["risk_chain"], "root:", plate["root_allergen_track_id"])
