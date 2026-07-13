"""Flask backend for the live TrackSense dashboard (Phase 12).

Serves the single-page dashboard (backend/static) and a small JSON API over one
DemoController. The heavy objects (RF model, pipeline) are created ONCE and
reused; a background thread advances the demo on a wall-clock timer for the live
animation, while /api/demo/step and /api/demo/run drive it synchronously (used
by tests and for a deterministic, thread-free run).

Endpoints
  GET  /                     dashboard SPA
  GET  /api/status           config + demo status
  GET  /api/snapshot         everything the UI needs in one poll
  GET  /api/objects          per-object risk state (risk desc)
  GET  /api/risk-map         {timestamp, objects:[...]}
  GET  /api/events           timeline
  GET  /api/alerts           downstream-risk alerts
  GET  /api/explanations     propagation-chain explanations
  POST /api/demo/start       {scenario?, speed?} start live auto-play
  POST /api/demo/pause       pause auto-play
  POST /api/demo/resume      resume auto-play
  POST /api/demo/step        advance one frame (sync)
  POST /api/demo/run         run to completion (sync)
  POST /api/demo/reset       {scenario?} reset
  POST /api/cleaning-event   {track_id?|class?} apply a cleaning action
  POST /api/live/start       start live YOLO risk mode (camera -> risk pipeline)
  POST /api/live/stop        stop live YOLO risk mode
  GET  /api/live/status      live-mode running/model/physical_verification state
  GET  /camera               standalone live camera view (visual)
  GET  /api/camera/stream    MJPEG stream (fans out live frames when live mode is on)

Mock demo mode and live YOLO risk mode share ONE snapshot contract: /api/snapshot
serves the live snapshot while live mode runs, else the mock snapshot. The webcam
is guarded by a single lock so the visual stream and the live-risk loop never open
it twice. Transport is simple polling; same origin serves API + SPA (no CORS).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import BACKEND_HOST, RF_MODEL_PATH, YOLO_MODEL_PATH, summary  # noqa: E402
from pipeline.demo_controller import DemoController  # noqa: E402
from pipeline.live_risk_service import CAMERA_LOCK, LiveRiskService  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = str(Path(__file__).resolve().parent / "static")
TEMPLATES_DIR = str(Path(__file__).resolve().parent / "templates")
SMOKE_DIR = str(PROJECT_ROOT / "reports" / "yolo_smoke")
SMOKE_SUMMARY_FILE = "yolo_smoke_summary.md"
SMOKE_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _list_smoke_images():
    """Image files in reports/yolo_smoke, judge-relevant first, capped. Never raises."""
    if not os.path.isdir(SMOKE_DIR):
        return []
    try:
        names = os.listdir(SMOKE_DIR)
    except OSError:
        return []
    files = [f for f in names
             if f.lower().endswith(SMOKE_IMAGE_EXTS) and not f.startswith("camdiag_")]

    def priority(name):
        low = name.lower()
        if low.startswith("smoke_whatsapp") or low.startswith("smoke_peanut"):
            return (0, low)          # annotated real still-image results (most relevant)
        if low.startswith("smoke_frame_"):
            return (3, low)          # camera-capture frames (least relevant here)
        if low.startswith("smoke_"):
            return (1, low)          # other annotated outputs
        return (2, low)              # raw source photos
    files.sort(key=priority)
    return files[:40]


# ---------------------------------------------------------------------------
# Live camera (server-side MJPEG stream) -- the multiscene 8-class detector.
# ---------------------------------------------------------------------------
# The camera detector is the configured 8-class model (defaults to the multiscene
# checkpoint; override with TRACKSENSE_YOLO_WEIGHTS). CAMERA_LOCK (shared with
# pipeline.live_risk_service) guarantees ONE physical camera handle across both
# the standalone visual stream and the live-risk loop.
CAMERA_MODEL_PATH = YOLO_MODEL_PATH
CAMERA_INDEX = int(os.environ.get("TRACKSENSE_CAMERA_INDEX", "0"))


def _mjpeg_stream(model, camera_index=0, jpeg_quality=70):
    """Standalone visual stream: yield annotated webcam frames as MJPEG. Releases
    BOTH the camera handle and CAMERA_LOCK when the client disconnects (tab closed).
    Only used when live-risk mode is NOT running (so the camera is never opened twice)."""
    import cv2  # lazy: only imported when a stream is actually requested

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # DSHOW is the reliable Windows backend
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    try:
        misses = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                misses += 1
                if misses > 90:      # camera stopped delivering frames -> give up
                    break
                continue
            misses = 0
            annotated = model(frame, verbose=False)[0].plot()  # boxes + labels drawn
            ok2, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok2:
                continue
            payload = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(payload)).encode() + b"\r\n\r\n" + payload + b"\r\n")
    finally:
        cap.release()
        try:
            CAMERA_LOCK.release()
        except RuntimeError:
            pass


def _shared_frame_stream(live_service):
    """Fan out the live-risk service's latest annotated frame -- NO new capture, so
    the visual stream and the live-risk loop share the one camera the service owns."""
    while live_service.running:
        payload = live_service.latest_jpeg()
        if payload:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(payload)).encode() + b"\r\n\r\n" + payload + b"\r\n")
        time.sleep(0.06)


class DemoRunner:
    """Thread-safe wrapper around a single DemoController."""

    def __init__(self, model_path=None):
        self._lock = threading.Lock()
        self._controller = DemoController(model_path=model_path)
        self._thread = None
        self._paused = True
        self._stop = False

    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop = False
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        while not self._stop:
            delay = 0.05
            with self._lock:
                if not self._paused and not self._controller.finished:
                    self._controller.step()
                    delay = self._controller.next_delay_seconds() or 0.05
            time.sleep(delay)

    # -- controls (each returns a fresh snapshot) --------------------------
    def start(self, scenario=None, speed=None):
        with self._lock:
            self._controller.reset(scenario)
            if speed is not None:
                self._controller.set_speed(speed)
            self._paused = False
            snap = self._controller.snapshot()
        self._ensure_thread()
        return snap

    def pause(self):
        with self._lock:
            self._paused = True
            return self._controller.snapshot()

    def resume(self):
        with self._lock:
            self._paused = False
            snap = self._controller.snapshot()
        self._ensure_thread()
        return snap

    def step(self):
        with self._lock:
            return self._controller.step()

    def run(self):
        with self._lock:
            return self._controller.run_to_completion()

    def reset(self, scenario=None):
        with self._lock:
            self._paused = True
            return self._controller.reset(scenario)

    def clean(self, track_id=None, class_name=None):
        with self._lock:
            if track_id is None and class_name is not None:
                for tid, st in self._controller.pipeline.engine.risk_map().items():
                    if st["class_name"] == class_name:
                        track_id = tid
                        break
            if track_id is None:
                return None, None
            prediction = self._controller.clean(track_id)
            return track_id, prediction

    def snapshot(self):
        with self._lock:
            return self._controller.snapshot()


def create_app(model_path=None) -> Flask:
    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static",
                template_folder=TEMPLATES_DIR)
    rf_path = model_path if model_path is not None else RF_MODEL_PATH
    app.config["RUNNER"] = DemoRunner(model_path=rf_path)
    app.config["LIVE"] = LiveRiskService(
        model_path=CAMERA_MODEL_PATH, rf_model_path=rf_path, camera_index=CAMERA_INDEX)

    def runner() -> DemoRunner:
        return app.config["RUNNER"]

    def live() -> LiveRiskService:
        return app.config["LIVE"]

    # -- SPA ---------------------------------------------------------------
    @app.route("/")
    def index():
        return render_template("index.html")

    # -- read endpoints ----------------------------------------------------
    @app.route("/api/status")
    def status():
        snap = runner().snapshot()
        return jsonify({
            "config": summary(),
            "detection_source": snap["source_kind"],
            "scenario": snap["scenario"],
            "status": snap["status"],
            "cursor": snap["cursor"],
            "total_frames": snap["total_frames"],
            "progress": snap["progress"],
            "timestamp": snap["timestamp"],
            "available_scenarios": snap["available_scenarios"],
        })

    @app.route("/api/snapshot")
    def snapshot():
        # When live YOLO risk mode is running, the dashboard reads its live snapshot;
        # otherwise it reads the mock demo snapshot (default).
        live_svc = live()
        if live_svc.running and live_svc.snapshot() is not None:
            return jsonify(live_svc.snapshot())
        return jsonify(runner().snapshot())

    @app.route("/api/objects")
    def objects():
        return jsonify({"objects": runner().snapshot()["objects"]})

    @app.route("/api/risk-map")
    def risk_map():
        snap = runner().snapshot()
        return jsonify({"timestamp": snap["timestamp"], "objects": snap["objects"]})

    @app.route("/api/events")
    def events():
        return jsonify({"events": runner().snapshot()["timeline"]})

    @app.route("/api/alerts")
    def alerts():
        return jsonify({"alerts": runner().snapshot()["alerts"]})

    @app.route("/api/explanations")
    def explanations():
        return jsonify({"explanations": runner().snapshot()["explanations"]})

    # -- demo controls -----------------------------------------------------
    @app.route("/api/demo/start", methods=["POST"])
    def demo_start():
        body = request.get_json(silent=True) or {}
        if live().running:            # switching to mock -> stop live first
            live().stop()
        return jsonify(runner().start(scenario=body.get("scenario"), speed=body.get("speed")))

    @app.route("/api/demo/pause", methods=["POST"])
    def demo_pause():
        return jsonify(runner().pause())

    @app.route("/api/demo/resume", methods=["POST"])
    def demo_resume():
        return jsonify(runner().resume())

    @app.route("/api/demo/step", methods=["POST"])
    def demo_step():
        return jsonify(runner().step())

    @app.route("/api/demo/run", methods=["POST"])
    def demo_run():
        return jsonify(runner().run())

    @app.route("/api/demo/reset", methods=["POST"])
    def demo_reset():
        body = request.get_json(silent=True) or {}
        if live().running:            # reset returns the dashboard to mock mode
            live().stop()
        return jsonify(runner().reset(scenario=body.get("scenario")))

    # -- cleaning ----------------------------------------------------------
    @app.route("/api/cleaning-event", methods=["POST"])
    def cleaning_event():
        body = request.get_json(silent=True) or {}
        track_id, prediction = runner().clean(
            track_id=body.get("track_id"), class_name=body.get("class"))
        if prediction is None:
            return jsonify({"ok": False, "error": "no matching tracked object to clean"}), 404
        return jsonify({"ok": True, "track_id": track_id, "prediction": prediction,
                        "snapshot": runner().snapshot()})

    # -- YOLO still-image smoke results (local, read-only) ------------------
    @app.route("/api/yolo-smoke-summary")
    def yolo_smoke_summary():
        summary_path = os.path.join(SMOKE_DIR, SMOKE_SUMMARY_FILE)
        exists = os.path.isfile(summary_path)
        text = ""
        if exists:
            try:
                with open(summary_path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                text = ""
        images = [{"filename": name, "url": "/reports/yolo_smoke/" + quote(name)}
                  for name in _list_smoke_images()]
        return jsonify({"summary_exists": exists, "summary_text": text, "images": images})

    @app.route("/reports/yolo_smoke/<path:filename>")
    def serve_smoke_file(filename):
        """Serve ONLY image/summary files from reports/yolo_smoke -- never arbitrary
        files. send_from_directory blocks path traversal; the extension allowlist
        blocks everything except the smoke images and the markdown summary."""
        allowed = SMOKE_IMAGE_EXTS + (".md",)
        if not filename.lower().endswith(allowed):
            abort(404)
        return send_from_directory(SMOKE_DIR, filename)

    # -- live YOLO risk mode (camera -> risk pipeline -> shared snapshot) ---
    @app.route("/api/live/start", methods=["POST"])
    def live_start():
        # Pause the mock auto-player so the two never both drive the snapshot.
        runner().pause()
        result = live().start()
        return jsonify(result), (200 if result.get("running") else 503)

    @app.route("/api/live/stop", methods=["POST"])
    def live_stop():
        return jsonify(live().stop())

    @app.route("/api/live/status")
    def live_status():
        return jsonify(live().status())

    # -- live camera (server-side YOLO, MJPEG) -----------------------------
    @app.route("/camera")
    def camera_page():
        return render_template("camera.html")

    @app.route("/api/camera/stream")
    def camera_stream():
        # If live-risk mode owns the camera, fan out ITS annotated frames (one capture).
        live_svc = live()
        if live_svc.running:
            return Response(_shared_frame_stream(live_svc),
                            mimetype="multipart/x-mixed-replace; boundary=frame")
        # Otherwise open a standalone capture. Load + schema-validate the detector once.
        try:
            model = app.config.get("CAMERA_MODEL")
            if model is None:
                from ultralytics import YOLO  # lazy heavy import
                from vision.yolo_detection_source import validate_class_names
                model = YOLO(CAMERA_MODEL_PATH)
                validate_class_names(model.names)   # reject the old / wrong-schema checkpoint
                app.config["CAMERA_MODEL"] = model
        except Exception as exc:  # noqa: BLE001 - surface load/schema errors to the client
            return jsonify({"error": "camera model load failed: " + str(exc)}), 500
        # One camera user at a time (single physical camera handle).
        if not CAMERA_LOCK.acquire(blocking=False):
            return jsonify({"error": "camera already in use (live mode or another tab)"}), 409
        return Response(_mjpeg_stream(model, CAMERA_INDEX),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


if __name__ == "__main__":
    # Flask's conventional dev port is 5000; honor TRACKSENSE_PORT if set.
    port = int(os.environ.get("TRACKSENSE_PORT", "5000"))
    application = create_app()
    print(f"TrackSense dashboard: http://{BACKEND_HOST}:{port}/")
    application.run(host=BACKEND_HOST, port=port, threaded=True, debug=False)
