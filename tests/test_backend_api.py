"""Flask backend API tests (Phase 17 items 15-17).

Builds the app against a small temp RF model (never the 49MB artifact) and drives
the demo synchronously via /api/demo/run and /api/demo/step so the tests are
deterministic and thread-free.

Run:  python -m unittest discover -s tests -p "test_backend_api.py"
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.generate_risk_training_data import generate_dataset  # noqa: E402
from ml.train_random_forest import parse_args, train  # noqa: E402
from backend.app import create_app  # noqa: E402

_TMP_DIR = None
_APP = None
_RF_MODEL = None


def setUpModule():
    global _TMP_DIR, _APP, _RF_MODEL
    _TMP_DIR = tempfile.mkdtemp(prefix="tracksense_api_")
    csv_path = os.path.join(_TMP_DIR, "risk_events.csv")
    model_path = os.path.join(_TMP_DIR, "rf.joblib")
    meta_path = os.path.join(_TMP_DIR, "rf.json")
    events_df, _ = generate_dataset(num_scenarios=300, seed=42)
    events_df.to_csv(csv_path, index=False)
    train(parse_args(["--no-search", "--n-estimators", "60", "--min-samples-leaf", "2",
                      "--data", csv_path, "--model-out", model_path,
                      "--metadata-out", meta_path, "--seed", "42"]))
    _RF_MODEL = model_path
    _APP = create_app(model_path=model_path)


def tearDownModule():
    if _TMP_DIR:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)


class TestBackendApi(unittest.TestCase):
    def setUp(self):
        self.c = _APP.test_client()
        self.c.post("/api/demo/reset", json={"scenario": "flagship_chain"})

    def test_index_and_static_served(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/static/app.js").status_code, 200)

    def test_status_endpoint(self):
        data = self.c.get("/api/status").get_json()
        self.assertEqual(data["detection_source"], "mock")
        self.assertIn("flagship_chain", data["available_scenarios"])
        self.assertIn("config", data)

    def test_demo_reset_is_idle_at_zero(self):
        data = self.c.post("/api/demo/reset", json={"scenario": "flagship_chain"}).get_json()
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["cursor"], 0)
        self.assertEqual(data["objects"], [])

    def test_step_advances_one_frame(self):
        before = self.c.get("/api/status").get_json()["cursor"]
        after = self.c.post("/api/demo/step").get_json()["cursor"]
        self.assertEqual(after, before + 1)

    def test_risk_map_after_run(self):
        self.c.post("/api/demo/run")
        rm = self.c.get("/api/risk-map").get_json()
        self.assertIn("timestamp", rm)
        classes = {o["class_name"] for o in rm["objects"]}
        self.assertIn("plate", classes)
        plate = next(o for o in rm["objects"] if o["class_name"] == "plate")
        self.assertEqual([len(plate["risk_chain"])], [4])  # jar->cutlery->bread->plate

    def test_events_and_alerts_after_run(self):
        self.c.post("/api/demo/run")
        events = self.c.get("/api/events").get_json()["events"]
        self.assertTrue(any(e["type"] == "source_detected" for e in events))
        self.assertTrue(any(e["type"] == "contact" for e in events))
        alerts = self.c.get("/api/alerts").get_json()["alerts"]
        self.assertTrue(all("chain_text" in a for a in alerts))

    def test_demo_start_returns_snapshot_then_pause(self):
        data = self.c.post("/api/demo/start", json={"scenario": "flagship_chain", "speed": 4}).get_json()
        self.assertIn("status", data)
        self.assertEqual(data["scenario"], "flagship_chain")
        self.c.post("/api/demo/pause")  # stop the background thread advancing

    def test_cleaning_event_endpoint(self):
        self.c.post("/api/demo/run")
        rm = self.c.get("/api/risk-map").get_json()
        cutlery = next(o for o in rm["objects"] if o["class_name"] == "cutlery")
        before = cutlery["risk_score"]
        res = self.c.post("/api/cleaning-event", json={"track_id": cutlery["track_id"]}).get_json()
        self.assertTrue(res["ok"])
        self.assertLessEqual(res["prediction"]["risk_score"], before)

    def test_cleaning_event_unknown_object_404(self):
        res = self.c.post("/api/cleaning-event", json={"track_id": 99999})
        self.assertEqual(res.status_code, 404)


class TestDashboardUiAndSmoke(unittest.TestCase):
    """The rebuilt demo UI page + the YOLO smoke-summary API (reuses _APP)."""

    def setUp(self):
        self.c = _APP.test_client()

    def test_index_returns_200_and_branding(self):
        resp = self.c.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("TrackSense", resp.get_data(as_text=True))

    def test_index_contains_physical_verification_flag(self):
        html = self.c.get("/").get_data(as_text=True)
        self.assertIn("physical_verification", html)
        self.assertIn("physical_verification: false", html)

    def test_index_contains_biochemical_disclaimer(self):
        html = self.c.get("/").get_data(as_text=True)
        self.assertIn(
            "Prototype relative-risk estimate. Not a biochemical allergen measurement.", html)

    def test_static_assets_served(self):
        self.assertEqual(self.c.get("/static/app.js").status_code, 200)
        self.assertEqual(self.c.get("/static/styles.css").status_code, 200)

    def test_yolo_smoke_summary_shape_with_or_without_file(self):
        data = self.c.get("/api/yolo-smoke-summary").get_json()
        self.assertIn("summary_exists", data)
        self.assertIsInstance(data["summary_exists"], bool)
        self.assertIn("summary_text", data)
        self.assertIsInstance(data["summary_text"], str)
        self.assertIsInstance(data["images"], list)
        for img in data["images"]:
            self.assertIn("filename", img)
            self.assertTrue(img["url"].startswith("/reports/yolo_smoke/"))

    def test_smoke_file_route_only_serves_allowed_extensions(self):
        # A non-image extension must never be served, even for a real file (404).
        self.assertEqual(self.c.get("/reports/yolo_smoke/app.py").status_code, 404)
        # A missing but allowed-extension file must 404 cleanly (no crash).
        self.assertEqual(self.c.get("/reports/yolo_smoke/does_not_exist.jpg").status_code, 404)

    def test_existing_api_endpoints_still_work(self):
        self.c.post("/api/demo/reset", json={"scenario": "flagship_chain"})
        self.c.post("/api/demo/run")
        for path in ("/api/status", "/api/snapshot", "/api/objects", "/api/risk-map",
                     "/api/events", "/api/alerts", "/api/explanations"):
            self.assertEqual(self.c.get(path).status_code, 200, path)
        snap = self.c.get("/api/snapshot").get_json()
        self.assertFalse(snap["physical_verification"])


_DUMMY_FRAME = object()   # detect_fn ignores frame content, so any sentinel works


class _FakeCap:
    """Fake OpenCV capture: yields sentinel frames (small sleep so the live thread
    keeps running while the test inspects it), never touches a real camera."""

    def __init__(self, max_frames=100000):
        self._max = max_frames
        self._i = 0

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.01)
        self._i += 1
        return (self._i <= self._max), _DUMMY_FRAME

    def release(self):
        pass


def _fake_detect(frame, frame_index, timestamp):
    """Synthetic detections (jar + cutlery, overlapping) -> exercises the real
    Detection -> tracker -> contact -> RF bridge with no YOLO/camera."""
    from pipeline.contracts import Detection
    jar = Detection.from_class_name("nut_butter_jar", 0.90, (10, 10, 90, 90), frame_index, timestamp)
    cutlery = Detection.from_class_name("cutlery", 0.85, (60, 60, 140, 140), frame_index, timestamp)
    return [jar, cutlery], None


class TestLiveRiskMode(unittest.TestCase):
    """Live YOLO risk mode via a mocked camera + detections (no hardware)."""

    def setUp(self):
        self.c = _APP.test_client()
        self._orig_live = _APP.config["LIVE"]

    def tearDown(self):
        try:
            _APP.config["LIVE"].stop()
        except Exception:
            pass
        _APP.config["LIVE"] = self._orig_live
        # return the dashboard to a clean mock state for other tests
        self.c.post("/api/demo/reset", json={"scenario": "flagship_chain"})

    def _install_fake_live(self):
        from pipeline.live_risk_service import LiveRiskService
        from pipeline.risk_pipeline import RiskPipeline
        svc = LiveRiskService(
            model_path="fake-model.pt", rf_model_path=_RF_MODEL, camera_index=0,
            capture_factory=_FakeCap, detect_fn=_fake_detect,
            pipeline_factory=lambda: RiskPipeline(model_path=_RF_MODEL, source_kind="yolo"))
        _APP.config["LIVE"] = svc
        return svc

    def test_live_status_returns_200_before_start(self):
        resp = self.c.get("/api/live/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["running"])
        self.assertEqual(data["source_kind"], "yolo/live")
        self.assertFalse(data["physical_verification"])

    def test_live_start_makes_snapshot_live_then_stop(self):
        svc = self._install_fake_live()
        started = self.c.post("/api/live/start")
        self.assertEqual(started.status_code, 200)
        self.assertTrue(started.get_json()["running"])

        for _ in range(300):                       # wait for the live loop to emit a snapshot
            if svc.snapshot() is not None:
                break
            time.sleep(0.01)
        self.assertIsNotNone(svc.snapshot())

        snap = self.c.get("/api/snapshot").get_json()   # /api/snapshot now serves LIVE
        self.assertEqual(snap.get("source_kind"), "yolo")
        self.assertFalse(snap["physical_verification"])
        for key in ("objects", "timeline", "alerts", "explanations"):
            self.assertIn(key, snap)

        status = self.c.get("/api/live/status").get_json()
        self.assertTrue(status["running"])

        self.assertFalse(self.c.post("/api/live/stop").get_json()["running"])
        self.assertFalse(svc.running)

    def test_mock_snapshot_when_live_off(self):
        # With live mode off, /api/snapshot stays on the mock pipeline.
        self.c.post("/api/demo/reset", json={"scenario": "flagship_chain"})
        self.c.post("/api/demo/run")
        snap = self.c.get("/api/snapshot").get_json()
        self.assertEqual(snap["source_kind"], "mock")
        self.assertFalse(snap["physical_verification"])
        self.assertTrue(any(o["class_name"] == "plate" for o in snap["objects"]))


if __name__ == "__main__":
    unittest.main()
