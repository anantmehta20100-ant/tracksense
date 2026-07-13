"""Runtime data contracts for the live TrackSense pipeline (Phase 2).

One source of truth for the four objects that flow through the runtime:

    Detection        -- one tracked object in one frame (detector/tracker output)
    ContactEvent     -- one meaningful contact between two tracked objects
    ObjectRiskState  -- the engine's current risk state for one tracked object
    RiskPrediction   -- the Random Forest's output for one contact

These are thin, validated dataclasses that *bridge* to the loose dict shapes the
existing modules already speak (vision/tracker.py detections, the
contact_event dict consumed by pipeline/risk_engine.py, and the RF prediction
dict from ml/risk_inference.py) via `.to_*()` / `.from_*()` helpers -- so the
older code keeps working unchanged while new code gets typed, range-checked
objects. Feature *order* still lives only in ml/risk_features.py; nothing here
duplicates it.

Units are documented on every field. Coordinates are pixels (image space);
times are seconds; risk scores are the RF's relative convenience score in [0,1].
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASS_TO_ID, get_allergen_type  # noqa: E402
from ml.risk_features import RISK_CLASS_LABELS, RISK_CLASS_TO_ID  # noqa: E402


class ContractError(ValueError):
    """Raised when a runtime object violates its contract (bad type/range)."""


def _check_bbox(bbox) -> Tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ContractError(f"bbox must have 4 values [x1,y1,x2,y2], got {bbox!r}")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 < x1 or y2 < y1:
        raise ContractError(f"bbox must be [x1,y1,x2,y2] with x2>=x1, y2>=y1, got {bbox!r}")
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """One detected + tracked object in one frame.

    track_id is None straight out of a raw detector and is filled in by the
    tracker (vision/tracker.py). class_id is the CANONICAL TrackSense id
    (config/allergens.OBJECT_CLASS_TO_ID), not a model-local id.
    """

    class_id: int
    class_name: str
    confidence: float           # [0,1]
    bbox_xyxy: Tuple[float, float, float, float]  # pixels
    frame_index: int
    timestamp: float            # seconds since capture start
    track_id: Optional[int] = None

    def __post_init__(self):
        self.bbox_xyxy = _check_bbox(self.bbox_xyxy)
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ContractError(f"confidence must be in [0,1], got {self.confidence}")
        self.confidence = float(self.confidence)
        self.frame_index = int(self.frame_index)
        self.timestamp = float(self.timestamp)

    # -- bridges to the existing loose dict shapes ------------------------
    def to_tracker_dict(self) -> dict:
        """The dict shape vision/tracker.IoUTracker.update() expects."""
        return {"class_name": self.class_name, "confidence": self.confidence,
                "bbox": list(self.bbox_xyxy)}

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id, "class_id": self.class_id,
            "class_name": self.class_name, "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy), "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
        }

    @classmethod
    def from_class_name(cls, class_name, confidence, bbox_xyxy, frame_index,
                        timestamp, track_id=None) -> "Detection":
        class_id = OBJECT_CLASS_TO_ID.get(class_name, -1)
        return cls(class_id=class_id, class_name=class_name, confidence=confidence,
                   bbox_xyxy=bbox_xyxy, frame_index=frame_index, timestamp=timestamp,
                   track_id=track_id)


# ---------------------------------------------------------------------------
# ContactEvent
# ---------------------------------------------------------------------------
@dataclass
class ContactEvent:
    """One meaningful, debounced contact between two tracked objects.

    Geometry (duration/overlap/distance) is direction-independent; source/target
    ordering is assigned by the pipeline from risk state (allergen or
    higher-current-risk object becomes the source). Use `.reversed()` to flip.
    """

    event_id: int
    source_track_id: int
    target_track_id: int
    source_class: str
    target_class: str
    start_time: float           # s, when the contact was first confirmed
    end_time: float             # s, when it ended (== start for still-active)
    duration: float             # s
    overlap_ratio: float        # bbox IoU-like [0,1]
    normalized_distance: float  # center gap / object size (0 == overlapping)
    frame_index: int            # frame the event was reported on
    timestamp: float            # s, report time
    repeated_contact_count: int = 0  # prior confirmed contacts for this ordered pair

    def reversed(self) -> "ContactEvent":
        return ContactEvent(
            event_id=self.event_id,
            source_track_id=self.target_track_id, target_track_id=self.source_track_id,
            source_class=self.target_class, target_class=self.source_class,
            start_time=self.start_time, end_time=self.end_time, duration=self.duration,
            overlap_ratio=self.overlap_ratio, normalized_distance=self.normalized_distance,
            frame_index=self.frame_index, timestamp=self.timestamp,
            repeated_contact_count=self.repeated_contact_count,
        )

    def to_engine_event(self) -> dict:
        """The contact_event dict shape pipeline/risk_engine.py consumes (same
        shape vision/contact_detector.py emits, so both engines are compatible)."""
        return {
            "source_track_id": self.source_track_id, "source_class": self.source_class,
            "target_track_id": self.target_track_id, "target_class": self.target_class,
            "timestamp": self.timestamp,
            "allergen_type": get_allergen_type(self.source_class),
        }

    def observations(self, cleaning_detected: int = 0) -> dict:
        """Measured feature observations for the RF feature builder (real values
        from geometry, not the engine's placeholder defaults)."""
        return {
            "contact_duration": self.duration,
            "bbox_overlap_ratio": self.overlap_ratio,
            "normalized_distance": self.normalized_distance,
            "cleaning_detected": int(cleaning_detected),
        }

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "source_track_id": self.source_track_id, "target_track_id": self.target_track_id,
            "source_class": self.source_class, "target_class": self.target_class,
            "start_time": round(self.start_time, 4), "end_time": round(self.end_time, 4),
            "duration": round(self.duration, 4), "overlap_ratio": round(self.overlap_ratio, 4),
            "normalized_distance": round(self.normalized_distance, 4),
            "frame_index": self.frame_index, "timestamp": round(self.timestamp, 4),
            "repeated_contact_count": self.repeated_contact_count,
        }


# ---------------------------------------------------------------------------
# RiskPrediction
# ---------------------------------------------------------------------------
@dataclass
class RiskPrediction:
    risk_class: str
    risk_class_id: int
    probabilities: dict         # {"LOW":.., "MEDIUM":.., "HIGH":..}
    risk_score: float           # continuous convenience score [0,1]
    model_version: str

    def __post_init__(self):
        if self.risk_class not in RISK_CLASS_LABELS:
            raise ContractError(f"risk_class must be one of {RISK_CLASS_LABELS}, got {self.risk_class!r}")
        if RISK_CLASS_TO_ID[self.risk_class] != self.risk_class_id:
            raise ContractError("risk_class / risk_class_id mismatch")

    @classmethod
    def from_dict(cls, d: dict) -> "RiskPrediction":
        return cls(risk_class=d["risk_class"], risk_class_id=d["risk_class_id"],
                   probabilities=dict(d["probabilities"]), risk_score=d["risk_score"],
                   model_version=d.get("model_version", "unknown"))

    def to_dict(self) -> dict:
        return {"risk_class": self.risk_class, "risk_class_id": self.risk_class_id,
                "probabilities": dict(self.probabilities),
                "risk_score": round(self.risk_score, 4), "model_version": self.model_version}


# ---------------------------------------------------------------------------
# ObjectRiskState
# ---------------------------------------------------------------------------
@dataclass
class ObjectRiskState:
    """The engine's per-object risk state. `risk_chain` is the ordered list of
    track_ids from the root allergen source down to this object (provenance)."""

    track_id: int
    class_name: str
    current_risk_score: float = 0.0
    current_risk_class: str = "LOW"
    current_risk_class_id: int = 0
    last_updated: Optional[float] = None
    contact_count: int = 0
    propagation_depth: int = 0
    source_exposure_time: Optional[float] = None
    parent_track_id: Optional[int] = None
    root_allergen_track_id: Optional[int] = None
    risk_chain: List[int] = field(default_factory=list)
    probabilities: dict = field(default_factory=lambda: {"LOW": 1.0, "MEDIUM": 0.0, "HIGH": 0.0})

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id, "class_name": self.class_name,
            "risk_score": round(self.current_risk_score, 4),
            "risk_class": self.current_risk_class,
            "risk_class_id": self.current_risk_class_id,
            "contact_count": self.contact_count,
            "propagation_depth": self.propagation_depth,
            "parent_track_id": self.parent_track_id,
            "root_allergen_track_id": self.root_allergen_track_id,
            "risk_chain": list(self.risk_chain),
            "last_updated": self.last_updated,
            "probabilities": dict(self.probabilities),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectRiskState":
        return cls(
            track_id=d["track_id"], class_name=d["class_name"],
            current_risk_score=d.get("risk_score", 0.0),
            current_risk_class=d.get("risk_class", "LOW"),
            current_risk_class_id=d.get("risk_class_id", 0),
            last_updated=d.get("last_updated"),
            contact_count=d.get("contact_count", 0),
            propagation_depth=d.get("propagation_depth", 0),
            parent_track_id=d.get("parent_track_id"),
            root_allergen_track_id=d.get("root_allergen_track_id"),
            risk_chain=list(d.get("risk_chain", [])),
            probabilities=dict(d.get("probabilities", {"LOW": 1.0, "MEDIUM": 0.0, "HIGH": 0.0})),
        )


if __name__ == "__main__":
    d = Detection.from_class_name("cutlery", 0.9, [10, 10, 90, 90], 0, 0.0)
    print("Detection:", d.to_dict())
    ev = ContactEvent(1, 3, 14, "nut_butter_jar", "cutlery", 1.0, 4.0, 3.0, 0.4, 0.1, 40, 4.0)
    print("ContactEvent:", ev.to_dict())
    print("engine event:", ev.to_engine_event())
    print("reversed src:", ev.reversed().source_class)
    print("contracts self-check PASS")
