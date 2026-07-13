"""Detects consumption events: a food object lingering near the user's mouth,
checked against the risk state and the session's stated allergen. This layer
is a heuristic consuming the GRU's risk output, not itself a trained model
(AGENTS.md "Consumption / exposure alert feature").
"""

import time

from config.allergens import (
    CONSUMPTION_PERSISTENCE_FRAMES,
    EXPOSURE_ALERT_RISK_THRESHOLD,
    FOOD_CLASSES,
    MOUTH_PROXIMITY_THRESHOLD_PX,
)


def _bbox_distance(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return (dx**2 + dy**2) ** 0.5


class ConsumptionTracker:
    def __init__(
        self,
        mouth_proximity_threshold_px: float = MOUTH_PROXIMITY_THRESHOLD_PX,
        persistence_frames: int = CONSUMPTION_PERSISTENCE_FRAMES,
        alert_threshold: float = EXPOSURE_ALERT_RISK_THRESHOLD,
    ):
        self.mouth_proximity_threshold_px = mouth_proximity_threshold_px
        self.persistence_frames = persistence_frames
        self.alert_threshold = alert_threshold

        self._proximity_streaks = {}  # track_id -> consecutive frames near mouth
        self._already_logged = set()  # track_ids logged during the current streak
        self.consumption_events = []  # full session log

    def update(self, tracks, mouth_bbox, risk_state, user_allergen: str):
        """tracks: list of vision.tracker.Track for the current frame.
        mouth_bbox: [x1, y1, x2, y2] from face_detector.detect(), or None.
        risk_state: pipeline.risk_state.RiskState instance.
        user_allergen: the session's stated allergen_type.

        Returns (new_events: list[dict], alert_triggered: bool).
        """
        if mouth_bbox is None:
            self._proximity_streaks.clear()
            self._already_logged.clear()
            return [], False

        new_events = []
        current_food_track_ids = set()

        for track in tracks:
            if track.class_name not in FOOD_CLASSES:
                continue
            current_food_track_ids.add(track.track_id)

            distance = _bbox_distance(track.bbox, mouth_bbox)
            if distance <= self.mouth_proximity_threshold_px:
                streak = self._proximity_streaks.get(track.track_id, 0) + 1
                self._proximity_streaks[track.track_id] = streak

                if streak >= self.persistence_frames and track.track_id not in self._already_logged:
                    new_events.append(self._log_event(track, risk_state, user_allergen))
                    self._already_logged.add(track.track_id)
            else:
                self._proximity_streaks.pop(track.track_id, None)
                self._already_logged.discard(track.track_id)

        stale_track_ids = [tid for tid in self._proximity_streaks if tid not in current_food_track_ids]
        for track_id in stale_track_ids:
            self._proximity_streaks.pop(track_id, None)
            self._already_logged.discard(track_id)

        alert_triggered = any(event["alert"] for event in new_events)
        return new_events, alert_triggered

    def _log_event(self, track, risk_state, user_allergen: str) -> dict:
        risk_at_time = risk_state.get_risk(track.track_id, user_allergen)
        event = {
            "object_id": track.track_id,
            "object_class": track.class_name,
            "allergen_type": user_allergen,
            "risk_at_time": risk_at_time,
            "timestamp": time.time(),
            "alert": risk_at_time >= self.alert_threshold,
        }
        self.consumption_events.append(event)
        return event
