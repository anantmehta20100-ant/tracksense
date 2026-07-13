"""Thin wrapper around an ultralytics YOLO model for kitchen object detection."""

import os

from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, weights_path: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"YOLO weights not found at '{weights_path}'.\n"
                "This model is produced by fine-tuning on the custom kitchen object "
                "classes (see AGENTS.md 'Object classes' and the Day 2-3 plan: collect "
                "training photos, then fine-tune YOLO on nut_butter_jar, whole_nuts, "
                "chopping_board, hand, etc.). Until that fine-tuned checkpoint exists, "
                "this wrapper has nothing to load."
            )
        self.model = YOLO(weights_path)

    def detect(self, frame):
        """Run detection on a single BGR frame (as from cv2.VideoCapture).

        Returns a list of {class_name, confidence, bbox: [x1, y1, x2, y2]}.
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return detections
