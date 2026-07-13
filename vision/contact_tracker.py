"""Stateful contact tracker with a debounced lifecycle (Phase 5).

The existing vision/contact_detector.py is a minimal proximity heuristic used by
the GRU live path: it emits a bare event once per streak and measures no
geometry. This module is the richer contact detector the Random Forest pipeline
needs. It is ADDITIVE -- contact_detector.py is left untouched.

Per object pair it runs a hysteresis state machine:

    NONE --(close for start_persistence frames)--> ACTIVE (contact confirmed)
    ACTIVE --(separated for end_persistence frames)--> ENDED  (event emitted)

and, while ACTIVE, it MEASURES the real features the RF wants:

    duration            = seconds from confirmation to separation
    overlap_ratio       = peak bbox IoU during the contact
    normalized_distance = closest edge gap / mean object size (0 == overlapping)

One physical interaction becomes exactly one ContactEvent (emitted on ENDED with
the fully-measured geometry), never one-per-frame. A short cooldown after a
contact ends stops a jittery pair from immediately re-emitting. Thresholds are
read from config/runtime_config.py (Phase 16). Contacts are emitted in track-id
order; the pipeline decides source/target orientation from risk state.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import CONTACT  # noqa: E402
from pipeline.contracts import ContactEvent  # noqa: E402
from vision.contact_detector import _bbox_distance  # reuse: edge gap in px  # noqa: E402
from vision.tracker import _iou  # reuse: IoU overlap  # noqa: E402

PairKey = Tuple[int, int]

NONE, PENDING, ACTIVE = "none", "pending", "active"


def _mean_box_size(box_a, box_b) -> float:
    def side(b):
        return ((b[2] - b[0]) + (b[3] - b[1])) / 2.0
    return max(1e-6, (side(box_a) + side(box_b)) / 2.0)


class _PairState:
    __slots__ = ("phase", "close_streak", "far_streak", "start_time", "start_frame",
                 "last_close_time", "peak_overlap", "min_norm_distance",
                 "cooldown_until", "confirmed_count")

    def __init__(self):
        self.phase = NONE
        self.close_streak = 0
        self.far_streak = 0
        self.start_time = None
        self.start_frame = None
        self.last_close_time = None
        self.peak_overlap = 0.0
        self.min_norm_distance = float("inf")
        self.cooldown_until = -1
        self.confirmed_count = 0

    def begin_active(self, timestamp, frame_index):
        self.phase = ACTIVE
        self.start_time = timestamp
        self.start_frame = frame_index
        self.last_close_time = timestamp
        self.far_streak = 0
        self.peak_overlap = 0.0
        self.min_norm_distance = float("inf")

    def reset_to_none(self):
        self.phase = NONE
        self.close_streak = 0
        self.far_streak = 0
        self.start_time = None
        self.start_frame = None
        self.last_close_time = None
        self.peak_overlap = 0.0
        self.min_norm_distance = float("inf")


class ContactTracker:
    def __init__(self, config=CONTACT):
        self.cfg = config
        self._pairs: Dict[PairKey, _PairState] = {}
        self._next_event_id = 1

    def reset(self):
        self._pairs.clear()
        self._next_event_id = 1

    # -- geometry ----------------------------------------------------------
    def _measure(self, track_a, track_b):
        gap = _bbox_distance(track_a.bbox, track_b.bbox)
        overlap = _iou(track_a.bbox, track_b.bbox)
        norm_distance = gap / _mean_box_size(track_a.bbox, track_b.bbox)
        close = (gap <= self.cfg.proximity_threshold_px) or (overlap > self.cfg.min_overlap_ratio)
        return close, overlap, norm_distance

    # -- main step ---------------------------------------------------------
    def update(self, tracks, frame_index: int, timestamp: float) -> List[ContactEvent]:
        """Advance every pair's lifecycle for this frame. Returns the list of
        ContactEvents for contacts that ENDED this frame (fully measured)."""
        by_id = {t.track_id: t for t in tracks}
        present_pairs = {}
        for track_a, track_b in combinations(tracks, 2):
            key = tuple(sorted((track_a.track_id, track_b.track_id)))
            present_pairs[key] = (track_a, track_b)

        ended: List[ContactEvent] = []
        # Consider every pair we currently see plus any we are mid-tracking.
        keys = set(present_pairs) | set(self._pairs)
        for key in keys:
            state = self._pairs.get(key)
            if state is None:
                state = self._pairs[key] = _PairState()

            pair = present_pairs.get(key)
            if pair is not None:
                close, overlap, norm_distance = self._measure(*pair)
            else:
                close, overlap, norm_distance = False, 0.0, float("inf")

            in_cooldown = frame_index < state.cooldown_until

            if state.phase == NONE:
                if close and not in_cooldown:
                    state.phase = PENDING
                    state.close_streak = 1
                # else stay NONE

            elif state.phase == PENDING:
                if close:
                    state.close_streak += 1
                    if state.close_streak >= self.cfg.start_persistence_frames:
                        state.begin_active(timestamp, frame_index)
                else:
                    state.reset_to_none()

            elif state.phase == ACTIVE:
                if close:
                    state.far_streak = 0
                    state.last_close_time = timestamp
                    state.peak_overlap = max(state.peak_overlap, overlap)
                    state.min_norm_distance = min(state.min_norm_distance, norm_distance)
                else:
                    state.far_streak += 1
                    if state.far_streak >= self.cfg.end_persistence_frames:
                        event = self._emit(key, pair or (by_id.get(key[0]), by_id.get(key[1])),
                                           state, frame_index, timestamp)
                        if event is not None:
                            ended.append(event)
                        state.cooldown_until = frame_index + self.cfg.cooldown_frames
                        state.confirmed_count += 1
                        state.reset_to_none()

        # Drop fully-idle pairs no longer on screen (keep cooldown/counted ones).
        for key in list(self._pairs):
            st = self._pairs[key]
            if (st.phase == NONE and key not in present_pairs
                    and frame_index >= st.cooldown_until and st.confirmed_count == 0):
                del self._pairs[key]
        return ended

    def _emit(self, key, pair, state, frame_index, timestamp) -> ContactEvent:
        track_a, track_b = pair
        if track_a is None or track_b is None:
            return None
        duration = max(0.0, (state.last_close_time or timestamp) - (state.start_time or timestamp))
        if duration < self.cfg.min_contact_duration_s:
            return None
        overlap = state.peak_overlap
        norm_distance = 0.0 if state.min_norm_distance == float("inf") else state.min_norm_distance
        event = ContactEvent(
            event_id=self._next_event_id,
            source_track_id=track_a.track_id, target_track_id=track_b.track_id,
            source_class=track_a.class_name, target_class=track_b.class_name,
            start_time=state.start_time or timestamp, end_time=state.last_close_time or timestamp,
            duration=duration, overlap_ratio=overlap, normalized_distance=norm_distance,
            frame_index=frame_index, timestamp=timestamp,
            repeated_contact_count=state.confirmed_count,
        )
        self._next_event_id += 1
        return event

    def flush(self, frame_index: int, timestamp: float, tracks=None) -> List[ContactEvent]:
        """Emit any still-ACTIVE contacts as ended (call at stream end)."""
        by_id = {t.track_id: t for t in (tracks or [])}
        ended: List[ContactEvent] = []
        for key, state in self._pairs.items():
            if state.phase == ACTIVE:
                pair = (by_id.get(key[0]), by_id.get(key[1]))
                event = self._emit(key, pair, state, frame_index, timestamp)
                if event is not None:
                    ended.append(event)
                state.confirmed_count += 1
                state.reset_to_none()
        return ended

    def active_contacts(self) -> List[dict]:
        """Lifecycle snapshot for the status view (pairs currently pending/active)."""
        out = []
        for key, state in self._pairs.items():
            if state.phase in (PENDING, ACTIVE):
                out.append({"pair": list(key), "phase": state.phase,
                            "close_streak": state.close_streak})
        return out


if __name__ == "__main__":
    from vision.mock_detection_source import MockDetectionSource
    from vision.tracker import IoUTracker

    src = MockDetectionSource("flagship_chain")
    tracker, contacts = IoUTracker(), ContactTracker()
    last_ts = 0.0
    for fd in src.frames():
        tracks = tracker.update([d.to_tracker_dict() for d in fd.detections])
        last_ts = fd.timestamp
        for ev in contacts.update(tracks, fd.frame_index, fd.timestamp):
            print(f"CONTACT {ev.source_class}<->{ev.target_class} "
                  f"dur={ev.duration:.2f}s overlap={ev.overlap_ratio:.3f} "
                  f"nd={ev.normalized_distance:.3f} rep={ev.repeated_contact_count}")
