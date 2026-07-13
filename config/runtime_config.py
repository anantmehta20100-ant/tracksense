"""Central runtime configuration for the live TrackSense integration pipeline.

Phase 16 of the integration build: one place for every knob the live flow
(detection source -> tracking -> contact detection -> feature builder -> Random
Forest -> RiskEngine -> Flask API -> dashboard) reads, so nothing is scattered
as a magic constant.

This module deliberately does NOT redefine the canonical object classes, allergen
sources, or the RF feature schema -- those stay in config/allergens.py,
ml/class_schema.py and ml/risk_features.py (their single sources of truth). It
only adds the *integration/runtime* settings and re-exports a few existing ones
for convenience.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import (  # noqa: E402
    CONTACT_PERSISTENCE_FRAMES,
    CONTACT_PROXIMITY_THRESHOLD_PX,
    EXPOSURE_ALERT_RISK_THRESHOLD,
)
from ml.class_schema import training_names  # noqa: E402

# ---------------------------------------------------------------------------
# Detection source selection
# ---------------------------------------------------------------------------
# "mock" -> vision.mock_detection_source.MockDetectionSource (works today)
# "yolo" -> vision.yolo_detection_source.YoloDetectionSource (needs 8-class best.pt)
DETECTION_SOURCE = os.environ.get("TRACKSENSE_DETECTION_SOURCE", "mock")

# Path where the FINAL trained 8-class detector should be dropped. Prefers the
# recovered MULTISCENE checkpoint (trained on multi-object scenes so it can
# co-detect e.g. nut_butter_jar + cutlery), overridable via TRACKSENSE_YOLO_WEIGHTS.
# Deliberately NOT model/checkpoints/best.pt (the old single-class cutlery model).
# The YOLO adapter validates class names on load and refuses a mismatched model,
# so an accidental old/wrong checkpoint fails fast rather than silently mislabelling.
YOLO_MODEL_PATH = os.environ.get(
    "TRACKSENSE_YOLO_WEIGHTS",
    os.path.join("model", "checkpoints", "tracksense_8class_multiscene_best.pt"),
)

# The class names the final detector MUST expose (model-local ids 0..7). Sourced
# from ml/class_schema.py so this never drifts from the trained schema.
EXPECTED_YOLO_CLASS_NAMES = training_names()  # {0: "nut_butter_jar", ..., 7: "bread"}

# RF model path. None -> use ml.train_random_forest.MODEL_PATH default.
RF_MODEL_PATH = os.environ.get("TRACKSENSE_RF_MODEL", None)

# ---------------------------------------------------------------------------
# Contact detection thresholds (vision/contact_tracker.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContactConfig:
    # Two objects are "close" when their bbox gap <= this many pixels OR they
    # overlap. Reuses the project-wide proximity threshold.
    proximity_threshold_px: float = float(CONTACT_PROXIMITY_THRESHOLD_PX)
    # Minimum bbox IoU-overlap to treat a pair as touching regardless of the gap.
    min_overlap_ratio: float = 0.0
    # A contact is only CONFIRMED (event emitted) after the pair stays close for
    # this many consecutive frames -- debounces detector jitter (hysteresis in).
    start_persistence_frames: int = int(CONTACT_PERSISTENCE_FRAMES)
    # A confirmed contact only ENDS after the pair stays separated this many
    # consecutive frames -- debounces flicker (hysteresis out).
    end_persistence_frames: int = 3
    # Minimum wall-clock duration (s) a contact must reach to be reported as a
    # completed ContactEvent. Filters instantaneous grazes.
    min_contact_duration_s: float = 0.0
    # After a pair's contact ends, ignore new contacts between them for this many
    # frames (prevents a jittery pair re-emitting back-to-back events).
    cooldown_frames: int = 2


CONTACT = ContactConfig()

# ---------------------------------------------------------------------------
# Risk alerting (dashboard / API)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlertConfig:
    # Raise a dashboard alert when an object's continuous risk score reaches this
    # (aligned with the existing exposure-alert threshold) OR its class is HIGH.
    alert_risk_score: float = float(EXPOSURE_ALERT_RISK_THRESHOLD)
    alert_on_high_class: bool = True
    # Score bands used only for coloring the UI (the RF's own LOW/MEDIUM/HIGH
    # class is authoritative; these just tint the score badge).
    medium_score: float = 0.30
    high_score: float = 0.60


ALERT = AlertConfig()

# ---------------------------------------------------------------------------
# Demo playback (pipeline/demo_controller.py, backend/app.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DemoConfig:
    scenario: str = "flagship_chain"
    fps: float = 10.0          # simulated capture rate of the mock source
    speed: float = 1.0         # >1 fast-forwards the animation (wall-clock only)
    seed: int = 42


DEMO = DemoConfig()

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
BACKEND_HOST = os.environ.get("TRACKSENSE_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("TRACKSENSE_PORT", "8000"))


def summary() -> dict:
    """Flat snapshot of the active configuration (surfaced by GET /api/status)."""
    return {
        "detection_source": DETECTION_SOURCE,
        "yolo_model_path": YOLO_MODEL_PATH,
        "rf_model_path": RF_MODEL_PATH,
        "expected_yolo_class_names": EXPECTED_YOLO_CLASS_NAMES,
        "contact": vars(CONTACT),
        "alert": vars(ALERT),
        "demo": vars(DEMO),
        "backend": {"host": BACKEND_HOST, "port": BACKEND_PORT},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(summary(), indent=2, default=str))
