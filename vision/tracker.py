"""Simple IoU-based multi-object tracker. Assigns consistent track IDs across
frames by matching each frame's detections to the previous frame's tracks by
bounding-box overlap. No motion model, no re-identification -- deliberately
simple, matches AGENTS.md's "explainable heuristic" design choice.
"""

from config.allergens import TRACKER_IOU_MATCH_THRESHOLD, TRACKER_MAX_MISSED_FRAMES


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class Track:
    def __init__(self, track_id, detection):
        self.track_id = track_id
        self.class_name = detection["class_name"]
        self.confidence = detection["confidence"]
        self.bbox = detection["bbox"]
        self.missed_frames = 0

    def update(self, detection):
        self.class_name = detection["class_name"]
        self.confidence = detection["confidence"]
        self.bbox = detection["bbox"]
        self.missed_frames = 0


class IoUTracker:
    def __init__(
        self,
        iou_threshold: float = TRACKER_IOU_MATCH_THRESHOLD,
        max_missed_frames: int = TRACKER_MAX_MISSED_FRAMES,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks = {}
        self._next_track_id = 0

    def update(self, detections):
        """Update tracks with the current frame's detections.

        Returns the list of currently active Track objects (after adding new
        tracks and dropping stale ones).
        """
        existing_track_ids = list(self.tracks.keys())
        unmatched_detections = list(range(len(detections)))
        matched_track_ids = set()

        # Greedy matching: highest IoU pairs first.
        candidate_pairs = []
        for track_id, track in self.tracks.items():
            for det_idx in unmatched_detections:
                iou = _iou(track.bbox, detections[det_idx]["bbox"])
                if iou >= self.iou_threshold:
                    candidate_pairs.append((iou, track_id, det_idx))
        candidate_pairs.sort(key=lambda pair: pair[0], reverse=True)

        used_det_indices = set()
        for iou, track_id, det_idx in candidate_pairs:
            if track_id in matched_track_ids or det_idx in used_det_indices:
                continue
            self.tracks[track_id].update(detections[det_idx])
            matched_track_ids.add(track_id)
            used_det_indices.add(det_idx)

        unmatched_detections = [i for i in unmatched_detections if i not in used_det_indices]

        # New tracks for unmatched detections.
        for det_idx in unmatched_detections:
            track_id = self._next_track_id
            self._next_track_id += 1
            self.tracks[track_id] = Track(track_id, detections[det_idx])

        # Age tracks that existed before this frame and went unmatched; drop
        # ones that have been missing too long.
        stale_track_ids = []
        for track_id in existing_track_ids:
            if track_id not in matched_track_ids:
                self.tracks[track_id].missed_frames += 1
                if self.tracks[track_id].missed_frames > self.max_missed_frames:
                    stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            del self.tracks[track_id]

        return list(self.tracks.values())
