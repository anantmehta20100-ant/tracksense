"""Final YOLO detection source (Phase 15) -- the plug-in point for `best.pt`.

Implements the SAME source interface as vision/mock_detection_source.py:
`frames()` yields FrameData(frame_index, timestamp, detections, control_events),
where detections are pipeline.contracts.Detection objects. So once the 8-class
model finishes training, switching from mock to real detection is a one-line
config change (TRACKSENSE_DETECTION_SOURCE=yolo) -- nothing downstream changes.

Safety rails required by the spec:
  * If the weights file is missing, fail with a clear message explaining how to
    supply it -- never silently fall back to the old single-class checkpoint at
    model/checkpoints/best.pt.
  * On load, VALIDATE the model's class names against the expected 8-class
    schema (ml/class_schema.training_names(): ids 0..7, bread==7, counter
    absent). A mismatch (e.g. the old single-class cutlery model) fails fast.

ultralytics is imported lazily so this module (and its name-validation helper)
can be imported/tested without loading a heavy model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASS_TO_ID  # noqa: E402
from config.runtime_config import EXPECTED_YOLO_CLASS_NAMES, YOLO_MODEL_PATH  # noqa: E402
from pipeline.contracts import Detection  # noqa: E402
from vision.mock_detection_source import FrameData  # reuse the frame container  # noqa: E402


class ModelSchemaMismatch(ValueError):
    """Raised when a loaded detector's classes are not the expected 8 classes."""


def _normalize_names(names) -> Dict[int, str]:
    """ultralytics exposes model.names as {int|str: str} or a list; normalize to
    {int: str}."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {int(i): str(v) for i, v in enumerate(names)}


def validate_class_names(model_names) -> Dict[int, str]:
    """Raise ModelSchemaMismatch unless `model_names` is exactly the expected
    8-class schema. Returns the normalized names on success."""
    got = _normalize_names(model_names)
    expected = {int(k): v for k, v in EXPECTED_YOLO_CLASS_NAMES.items()}
    if got != expected:
        raise ModelSchemaMismatch(
            "Loaded detector class names do not match the TrackSense 8-class schema.\n"
            f"  expected: {expected}\n"
            f"  got:      {got}\n"
            "Refusing to use this checkpoint. This most often means an OLD single-class "
            "checkpoint (e.g. the cutlery-only model/checkpoints/best.pt) was supplied "
            "instead of the final 8-class detector. Point TRACKSENSE_YOLO_WEIGHTS at the "
            "correct weights. Expected names come from ml/class_schema.training_names()."
        )
    return got


class YoloDetectionSource:
    """Wraps an ultralytics YOLO model and emits Detection objects."""

    def __init__(self, model_path: str = None, *, fps: float = 30.0, camera_index: int = 0,
                 video_path: str = None):
        self.model_path = model_path or YOLO_MODEL_PATH
        self.fps = float(fps)
        self.camera_index = camera_index
        self.video_path = video_path
        self.names = self._load()

    @property
    def source_kind(self) -> str:
        return "yolo"

    def _load(self) -> Dict[int, str]:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"YOLO weights not found at '{self.model_path}'.\n"
                "The final 8-class detector is still training separately (Kaggle). When it "
                "finishes, save it to this path (or set TRACKSENSE_YOLO_WEIGHTS) and switch "
                "TRACKSENSE_DETECTION_SOURCE=yolo. Do NOT reuse model/checkpoints/best.pt if "
                "it is the old single-class cutlery model -- the class-name check will reject it."
            )
        from ultralytics import YOLO  # lazy: only needed when actually loading weights

        self.model = YOLO(self.model_path)
        return validate_class_names(self.model.names)

    def detect(self, frame, frame_index: int = 0, timestamp: float = 0.0):
        """Run the detector on one BGR frame -> list[Detection] (canonical classes)."""
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            local_id = int(box.cls[0])
            class_name = self.names.get(local_id, str(local_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                class_id=OBJECT_CLASS_TO_ID.get(class_name, -1), class_name=class_name,
                confidence=max(0.0, min(1.0, confidence)), bbox_xyxy=(x1, y1, x2, y2),
                frame_index=frame_index, timestamp=timestamp))
        return detections

    def frames(self):
        """Yield FrameData from a camera or video file. control_events is always
        empty for YOLO -- cleaning is a manual runtime event (Phase 10), and
        allergen 'sources' come from detected classes, not a scripted answer key."""
        import cv2  # lazy import; only needed for a real capture

        source = self.video_path if self.video_path is not None else self.camera_index
        capture = cv2.VideoCapture(source)
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / self.fps
                yield FrameData(frame_index=frame_index, timestamp=timestamp,
                                detections=self.detect(frame, frame_index, timestamp),
                                control_events=[])
                frame_index += 1
        finally:
            capture.release()


def build_detection_source(kind: str = None, **kwargs):
    """Factory used by the backend: return a mock or YOLO source by kind."""
    from config.runtime_config import DETECTION_SOURCE

    kind = (kind or DETECTION_SOURCE).lower()
    if kind == "yolo":
        return YoloDetectionSource(**kwargs)
    from vision.mock_detection_source import MockDetectionSource

    allowed = {k: kwargs[k] for k in ("scenario", "fps", "seed") if k in kwargs}
    return MockDetectionSource(**allowed)


if __name__ == "__main__":
    # No weights required: demonstrate the fail-fast class validation.
    print("expected 8-class names:", EXPECTED_YOLO_CLASS_NAMES)
    try:
        validate_class_names({0: "cutlery"})  # the old single-class model
    except ModelSchemaMismatch as exc:
        print("\nrejected old single-class model as expected:\n", str(exc).splitlines()[0])
    ok = validate_class_names(EXPECTED_YOLO_CLASS_NAMES)
    print("\naccepted correct 8-class schema:", ok)
