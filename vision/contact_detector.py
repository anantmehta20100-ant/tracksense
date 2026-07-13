"""Proximity/overlap heuristic that turns tracked objects into contact events.

Checks every pair of currently tracked objects each frame. If a pair's
bounding boxes overlap or stay within CONTACT_PROXIMITY_THRESHOLD_PX of each
other for CONTACT_PERSISTENCE_FRAMES consecutive frames, a contact event is
emitted once (not re-emitted every frame while still touching).
"""

import time
from itertools import combinations

from config.allergens import (
    CONTACT_PERSISTENCE_FRAMES,
    CONTACT_PROXIMITY_THRESHOLD_PX,
    get_allergen_type,
)


def _bbox_distance(box_a, box_b):
    """Gap between two bboxes in pixels. 0 (or negative, clamped to 0) if
    they overlap."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return (dx**2 + dy**2) ** 0.5


class ContactDetector:
    def __init__(
        self,
        proximity_threshold_px: float = CONTACT_PROXIMITY_THRESHOLD_PX,
        persistence_frames: int = CONTACT_PERSISTENCE_FRAMES,
    ):
        self.proximity_threshold_px = proximity_threshold_px
        self.persistence_frames = persistence_frames
        # (track_id_a, track_id_b) -> consecutive frames within threshold
        self._proximity_streaks = {}
        # pairs that have already fired a contact event during the current streak
        self._already_emitted = set()

    def update(self, tracks):
        """tracks: list of vision.tracker.Track objects for the current frame.

        Returns a list of new contact events:
        {source_track_id, source_class, target_track_id, target_class,
         timestamp, allergen_type}
        """
        events = []
        current_pairs = set()

        for track_a, track_b in combinations(tracks, 2):
            pair_key = tuple(sorted((track_a.track_id, track_b.track_id)))
            current_pairs.add(pair_key)

            distance = _bbox_distance(track_a.bbox, track_b.bbox)
            if distance <= self.proximity_threshold_px:
                streak = self._proximity_streaks.get(pair_key, 0) + 1
                self._proximity_streaks[pair_key] = streak

                if streak >= self.persistence_frames and pair_key not in self._already_emitted:
                    events.append(self._build_event(track_a, track_b))
                    self._already_emitted.add(pair_key)
            else:
                self._proximity_streaks.pop(pair_key, None)
                self._already_emitted.discard(pair_key)

        # Clean up state for pairs that no longer co-occur (a track dropped).
        stale_pairs = [p for p in self._proximity_streaks if p not in current_pairs]
        for pair_key in stale_pairs:
            self._proximity_streaks.pop(pair_key, None)
            self._already_emitted.discard(pair_key)

        return events

    @staticmethod
    def _build_event(track_a, track_b):
        # If one side of the pair is an allergen source class, treat it as
        # the "source" of the event; otherwise order is arbitrary (a/b).
        allergen_a = get_allergen_type(track_a.class_name)
        allergen_b = get_allergen_type(track_b.class_name)

        if allergen_a is not None:
            source, target, allergen_type = track_a, track_b, allergen_a
        elif allergen_b is not None:
            source, target, allergen_type = track_b, track_a, allergen_b
        else:
            source, target, allergen_type = track_a, track_b, None

        return {
            "source_track_id": source.track_id,
            "source_class": source.class_name,
            "target_track_id": target.track_id,
            "target_class": target.class_name,
            "timestamp": time.time(),
            "allergen_type": allergen_type,
        }
