"""Main live loop: camera -> object detection -> tracking -> contact
detection -> GRU risk update -> face detection -> consumption check.

Exposes the current risk state and active alerts via LiveRunner's own
attributes (self.risk_state, self.active_alerts) rather than a queue or
separate process. The dashboard (dashboard/app.py) holds one LiveRunner
instance in st.session_state and calls process_frame() once per rerun --
simplest option given everything already runs in a single local process and
AGENTS.md rules out any server-side state. Revisit if the dashboard ever
needs to run in a separate process from the capture loop.
"""

import argparse
import os

import cv2

from config.allergens import ALLERGEN_TYPES
from pipeline.consumption import ConsumptionTracker
from pipeline.risk_state import DEFAULT_CHECKPOINT_PATH, RiskState
from vision.contact_detector import ContactDetector
from vision.face_detector import FaceDetector
from vision.object_detector import ObjectDetector
from vision.tracker import IoUTracker

DEFAULT_YOLO_WEIGHTS_PATH = os.path.join("model", "checkpoints", "yolo_kitchen.pt")


class LiveRunner:
    def __init__(
        self,
        user_allergen: str,
        weights_path: str = DEFAULT_YOLO_WEIGHTS_PATH,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        camera_index: int = 0,
    ):
        self.user_allergen = user_allergen
        self.camera_index = camera_index

        self.object_detector = ObjectDetector(weights_path)
        self.face_detector = FaceDetector()
        self.tracker = IoUTracker()
        self.contact_detector = ContactDetector()
        self.risk_state = RiskState(checkpoint_path)
        self.consumption_tracker = ConsumptionTracker()

        self.capture = None
        self.active_alerts = []

    def start_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.camera_index)

    def stop_camera(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def process_frame(self):
        """Reads and processes one frame. Returns a dict with the frame,
        current tracks, mouth bbox, live risk_state, and active_alerts, or
        None if no frame was available (e.g. camera not opened / stream end)."""
        self.start_camera()
        success, frame = self.capture.read()
        if not success:
            return None

        detections = self.object_detector.detect(frame)
        tracks = self.tracker.update(detections)

        for contact_event in self.contact_detector.update(tracks):
            self.risk_state.update(contact_event)

        mouth_bbox = self.face_detector.detect(frame)
        new_events, alert_triggered = self.consumption_tracker.update(
            tracks, mouth_bbox, self.risk_state, self.user_allergen
        )
        if alert_triggered:
            self.active_alerts.extend(event for event in new_events if event["alert"])

        return {
            "frame": frame,
            "tracks": tracks,
            "mouth_bbox": mouth_bbox,
            "risk_state": self.risk_state,
            "active_alerts": list(self.active_alerts),
        }

    def run_cli(self):
        """Standalone debug loop with an OpenCV preview window (no
        Streamlit). Press 'q' to quit."""
        self.start_camera()
        try:
            while True:
                result = self.process_frame()
                if result is None:
                    break

                frame = result["frame"]
                for track in result["tracks"]:
                    x1, y1, x2, y2 = (int(v) for v in track.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{track.class_name}#{track.track_id}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                if result["active_alerts"]:
                    cv2.putText(
                        frame,
                        "EXPOSURE ALERT",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("TrackSense (debug)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.stop_camera()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TrackSense live pipeline standalone (debug view).")
    parser.add_argument("--user-allergen", required=True, choices=ALLERGEN_TYPES)
    parser.add_argument("--weights", default=DEFAULT_YOLO_WEIGHTS_PATH)
    parser.add_argument("--gru-checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    runner = LiveRunner(
        user_allergen=args.user_allergen,
        weights_path=args.weights,
        checkpoint_path=args.gru_checkpoint,
        camera_index=args.camera_index,
    )
    runner.run_cli()
