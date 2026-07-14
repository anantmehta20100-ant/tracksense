"""Live YOLO -> risk service for the dashboard backend.

One background thread owns ONE webcam capture and, per frame:

    capture frame
      -> YOLO (multiscene 8-class detector)
      -> pipeline.contracts.Detection objects
      -> RiskPipeline.process_frame()  (IoUTracker -> ContactTracker -> RF -> RiskEngine)
      -> latest live snapshot (served by GET /api/snapshot when live mode is on)
      -> latest annotated JPEG (fanned out to the visual /api/camera/stream)

It reuses the EXACT bridge the CLI runner uses (pipeline.live_yolo_runner.
_detections_from_result + RiskPipeline(source_kind="yolo")) so live risk numbers
come only from real detections through the real contact tracker + Random Forest --
nothing fabricates risk, and physical_verification stays False.

Single-camera rule: `CAMERA_LOCK` is the one arbiter of the physical webcam. The
live service holds it while running; the standalone visual stream in backend/app.py
uses the SAME lock, so the two never open the camera twice. When live mode is on,
the visual stream fans out this service's annotated frames instead of opening its own.

Testability: `capture_factory`, `detect_fn`, and `pipeline_factory` are injectable,
so tests drive the real bridge with a fake capture + synthetic detections and never
touch a real camera or ultralytics.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# One arbiter for the single physical camera handle (shared with backend/app.py).
CAMERA_LOCK = threading.Lock()

_LIVE_SCENARIOS = ["live_yolo"]


class LiveRiskService:
    """Runs (or simulates) a live YOLO -> risk loop in a background thread."""

    def __init__(self, *, model_path, rf_model_path=None, camera_index=0,
                 contact_config=None, imgsz=448, jpeg_quality=70,
                 capture_factory=None, detect_fn=None, pipeline_factory=None):
        self.model_path = str(model_path)
        self.rf_model_path = rf_model_path
        self.camera_index = int(camera_index)
        self._contact_config = contact_config
        self._imgsz = int(imgsz)
        self._jpeg_quality = int(jpeg_quality)

        # Injectable seams (defaults use the real camera + YOLO):
        self._capture_factory = capture_factory
        self._detect_fn = detect_fn
        self._pipeline_factory = pipeline_factory

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._running = False
        self._error = None

        self._model = None
        self._names = None
        self._detect = None
        self._pipeline = None

        self._latest_snapshot = None
        self._latest_jpeg = None
        self._last_frame_index = -1
        self._last_update_time = None

    # -- public state ------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    def snapshot(self):
        return self._latest_snapshot

    def latest_jpeg(self):
        return self._latest_jpeg

    def status(self) -> dict:
        return {
            "running": bool(self._running),
            "source_kind": "yolo/live",
            "model_path": self.model_path,
            "physical_verification": False,   # never physically validated
            "last_frame_index": self._last_frame_index,
            "last_update_time": self._last_update_time,
            "error": self._error,
        }

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> dict:
        with self._lock:
            if self._running:
                return {"running": True, "already_running": True}

            # Build the detector + pipeline (fail fast on a missing/wrong model).
            try:
                if self._detect_fn is None:
                    from ultralytics import YOLO  # lazy heavy import
                    from vision.yolo_detection_source import validate_class_names
                    self._model = YOLO(self.model_path)
                    self._names = validate_class_names(self._model.names)  # rejects wrong schema
                    self._detect = self._default_detect
                else:
                    self._detect = self._detect_fn

                if self._pipeline_factory is not None:
                    self._pipeline = self._pipeline_factory()
                else:
                    from config.runtime_config import CONTACT
                    from pipeline.risk_pipeline import RiskPipeline
                    self._pipeline = RiskPipeline(
                        model_path=self.rf_model_path,
                        contact_config=self._contact_config or CONTACT,
                        source_kind="yolo")
            except Exception as exc:  # noqa: BLE001 - surface load/schema errors to caller
                self._error = str(exc)
                return {"running": False, "error": str(exc)}

            # One physical camera at a time.
            if not CAMERA_LOCK.acquire(blocking=False):
                self._error = "camera busy (a camera tab may be open); close it and retry"
                return {"running": False, "error": self._error}

            self._stop.clear()
            self._error = None
            self._latest_snapshot = None
            self._last_frame_index = -1
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return {"running": True}

    def stop(self) -> dict:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        self._running = False
        self._release_lock()
        # Flush any still-active contact and finalize the snapshot as "stopped".
        if self._pipeline is not None:
            try:
                self._pipeline.finish()
                snap = self._decorate(self._pipeline.snapshot(), status="stopped")
                self._latest_snapshot = snap
            except Exception:  # noqa: BLE001 - stop must never raise
                pass
        return {"running": False}

    # -- internals ---------------------------------------------------------
    def _release_lock(self):
        try:
            CAMERA_LOCK.release()
        except RuntimeError:
            pass

    def _decorate(self, snap: dict, *, status: str, frame_index: int = None) -> dict:
        """Add the dashboard-contract fields the UI expects (mirrors DemoController)."""
        snap.update({
            "scenario": "live_yolo",
            "mode": "live",
            "status": status,
            "cursor": self._last_frame_index if frame_index is None else frame_index,
            "total_frames": 0,
            "progress": 0.0,
            "fps": 0.0,
            "speed": 1.0,
            "available_scenarios": _LIVE_SCENARIOS,
        })
        return snap

    def _default_capture(self):
        import cv2
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # reliable Windows backend
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab the freshest frame
        except Exception:  # noqa: BLE001
            pass
        return cap

    def _default_detect(self, frame, frame_index, timestamp):
        """Real YOLO detection -> (Detection list, annotated jpeg bytes)."""
        import cv2

        from pipeline.live_yolo_runner import (  # reuse the CLI bridge
            _detections_from_result, draw_collision_overlay, inflate_result_boxes)
        result = self._model(frame, verbose=False, imgsz=self._imgsz)[0]
        dets = _detections_from_result(result, self._names, frame_index, timestamp)
        inflate_result_boxes(result)  # visual only: enlarge drawn boxes (dets already extracted)
        annotated = draw_collision_overlay(result.plot(), result)  # flag touching boxes
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        return dets, (buf.tobytes() if ok else None)

    def _loop(self):
        from vision.mock_detection_source import FrameData

        try:
            cap = (self._capture_factory() if self._capture_factory else self._default_capture())
        except Exception as exc:  # noqa: BLE001
            self._error = "camera open failed: " + str(exc)
            self._running = False
            self._release_lock()
            return
        if cap is None or (hasattr(cap, "isOpened") and not cap.isOpened()):
            self._error = "could not open camera"
            self._running = False
            try:
                if cap is not None:
                    cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._release_lock()
            return

        frame_index = 0
        misses = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    misses += 1
                    if misses > 90:
                        self._error = "camera stopped delivering frames"
                        break
                    continue
                misses = 0
                timestamp = frame_index / 30.0
                try:
                    dets, jpeg = self._detect(frame, frame_index, timestamp)
                except Exception as exc:  # noqa: BLE001
                    self._error = "detection failed: " + str(exc)
                    break
                fd = FrameData(frame_index=frame_index, timestamp=timestamp,
                               detections=dets, control_events=[])
                self._pipeline.process_frame(fd)
                self._latest_snapshot = self._decorate(
                    self._pipeline.snapshot(), status="running", frame_index=frame_index)
                if jpeg:
                    self._latest_jpeg = jpeg
                self._last_frame_index = frame_index
                self._last_update_time = time.time()
                frame_index += 1
        finally:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._running = False
            self._release_lock()
