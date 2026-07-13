"""Thin wrapper around MediaPipe Face Detection for locating the mouth region."""

import mediapipe as mp

# Approximate fraction of the face bbox height (from the top) where the
# mouth region begins. MediaPipe's short-range face detector returns a
# whole-face bbox, not facial landmarks, so the mouth region is estimated as
# the lower portion of that box rather than detected directly.
MOUTH_REGION_TOP_FRACTION = 0.6


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        self._mp_face_detection = mp.solutions.face_detection
        self.detector = self._mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame):
        """Run face detection on a single BGR frame.

        Returns the estimated mouth region bbox [x1, y1, x2, y2] in pixel
        coordinates if a face is found, else None.
        """
        height, width = frame.shape[:2]
        rgb_frame = frame[:, :, ::-1]
        results = self.detector.process(rgb_frame)

        if not results.detections:
            return None

        # Use the highest-confidence detection.
        detection = max(results.detections, key=lambda d: d.score[0])
        box = detection.location_data.relative_bounding_box

        x1 = box.xmin * width
        y1 = box.ymin * height
        x2 = x1 + box.width * width
        y2 = y1 + box.height * height

        mouth_y1 = y1 + (y2 - y1) * MOUTH_REGION_TOP_FRACTION
        return [x1, mouth_y1, x2, y2]
