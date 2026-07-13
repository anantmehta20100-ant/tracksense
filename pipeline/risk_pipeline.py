"""End-to-end live risk pipeline (integration spine).

Wires the whole flow that the milestone is about, source-agnostic:

    FrameData (mock OR yolo)
      -> IoUTracker            (persistent track ids; reused as-is)
      -> ContactTracker        (debounced contacts w/ measured geometry)
      -> orient by risk state  (allergen / higher-risk object == source)
      -> RiskFeatureBuilder    (via the engine)
      -> Random Forest         (real inference)
      -> RiskEngine            (per-object risk state + propagation chain)
      -> snapshot()            (objects / alerts / explanations / timeline)

This is the RF analogue of pipeline/live_runner.py (which drives the GRU stack);
it does not touch that path. A FrameData is anything with `.detections`
(list of pipeline.contracts.Detection) and `.control_events` (list of dicts),
which both vision/mock_detection_source.py and vision/yolo_detection_source.py
produce -- so switching mock<->yolo needs no change here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import get_allergen_type  # noqa: E402
from config.runtime_config import CONTACT, YOLO_MODEL_PATH  # noqa: E402
from pipeline.propagation import build_alerts, build_explanations  # noqa: E402
from pipeline.risk_engine import RiskEngine  # noqa: E402
from vision.contact_tracker import ContactTracker  # noqa: E402
from vision.tracker import IoUTracker  # noqa: E402


def orient_contact(engine: RiskEngine, contact):
    """Decide which object is the contamination SOURCE for a geometric contact.

    Rule (contamination flows from more- to less-contaminated):
      1. a raw allergen source class is always the source;
      2. otherwise the object with the higher current risk is the source;
      3. ties -> the lower track id (deterministic).
    The ContactTracker emits pairs in ascending track-id order, so `contact`
    already has the lower id as source (used for tie-breaking).
    """
    a_allergen = get_allergen_type(contact.source_class) is not None
    b_allergen = get_allergen_type(contact.target_class) is not None
    if a_allergen and not b_allergen:
        return contact
    if b_allergen and not a_allergen:
        return contact.reversed()

    a = engine.get_risk(contact.source_track_id)
    b = engine.get_risk(contact.target_track_id)
    a_risk = a["risk_score"] if a else 0.0
    b_risk = b["risk_score"] if b else 0.0
    if b_risk > a_risk:
        return contact.reversed()
    return contact


class RiskPipeline:
    def __init__(self, model_path=None, contact_config=CONTACT, source_kind: str = "mock"):
        self.tracker = IoUTracker()
        self.contacts = ContactTracker(contact_config)
        self.engine = RiskEngine(model_path=model_path)
        self.source_kind = source_kind
        self.frame_index = -1
        self.timestamp = 0.0
        self.tracks = []
        self.timeline = []
        self._seen_sources = set()

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        self.tracker = IoUTracker()
        self.contacts.reset()
        self.engine.reset()
        self.frame_index = -1
        self.timestamp = 0.0
        self.tracks = []
        self.timeline = []
        self._seen_sources = set()

    # -- per-frame ---------------------------------------------------------
    def process_frame(self, frame_data) -> list:
        tracker_dets = [d.to_tracker_dict() for d in frame_data.detections]
        self.tracks = self.tracker.update(tracker_dets)
        self.frame_index = frame_data.frame_index
        self.timestamp = frame_data.timestamp

        results = []
        self._note_new_sources(frame_data.timestamp)

        for contact in self.contacts.update(self.tracks, frame_data.frame_index, frame_data.timestamp):
            results.append(self._handle_contact(contact))

        for control in getattr(frame_data, "control_events", []):
            res = self._handle_control(control, frame_data.timestamp)
            if res is not None:
                results.append(res)
        return results

    def finish(self):
        """Emit any still-active contacts at stream end."""
        results = []
        for contact in self.contacts.flush(self.frame_index, self.timestamp, self.tracks):
            results.append(self._handle_contact(contact))
        return results

    # -- internals ---------------------------------------------------------
    def _note_new_sources(self, timestamp):
        for track in self.tracks:
            if get_allergen_type(track.class_name) is not None and track.track_id not in self._seen_sources:
                self._seen_sources.add(track.track_id)
                self.timeline.append({
                    "type": "source_detected", "timestamp": round(timestamp, 3),
                    "frame_index": self.frame_index, "track_id": track.track_id,
                    "class": track.class_name,
                })

    def _handle_contact(self, contact):
        directed = orient_contact(self.engine, contact)
        observations = directed.observations()
        prediction = self.engine.process_contact_event(directed.to_engine_event(), observations=observations)
        self.timeline.append({
            "type": "contact", "timestamp": round(directed.timestamp, 3),
            "frame_index": directed.frame_index, "event_id": directed.event_id,
            "source_class": directed.source_class, "target_class": directed.target_class,
            "source_track_id": directed.source_track_id, "target_track_id": directed.target_track_id,
            "duration": round(directed.duration, 3),
            "overlap_ratio": round(directed.overlap_ratio, 4),
            "normalized_distance": round(directed.normalized_distance, 4),
            "risk_class": prediction["risk_class"], "risk_score": prediction["risk_score"],
        })
        return prediction

    def _handle_control(self, control, timestamp):
        if control.get("type") != "cleaning":
            return None
        track_id = self._resolve_track_by_class(control.get("class"))
        if track_id is None:
            return None
        prediction = self.engine.mark_cleaned(track_id, timestamp=timestamp)
        if prediction is None:
            return None
        self.timeline.append({
            "type": "cleaning", "timestamp": round(timestamp, 3),
            "frame_index": self.frame_index, "track_id": track_id,
            "class": control.get("class"),
            "risk_class": prediction["risk_class"], "risk_score": prediction["risk_score"],
        })
        return prediction

    def clean_track(self, track_id, timestamp=None):
        """Manual cleaning trigger (backend POST /api/cleaning-event)."""
        ts = timestamp if timestamp is not None else self.timestamp
        prediction = self.engine.mark_cleaned(track_id, timestamp=ts)
        if prediction is not None:
            state = self.engine.get_risk(track_id)
            self.timeline.append({
                "type": "cleaning", "timestamp": round(ts, 3), "frame_index": self.frame_index,
                "track_id": track_id, "class": state["class_name"] if state else None,
                "risk_class": prediction["risk_class"], "risk_score": prediction["risk_score"],
            })
        return prediction

    def _resolve_track_by_class(self, class_name):
        if class_name is None:
            return None
        for track in self.tracks:
            if track.class_name == class_name:
                return track.track_id
        # fall back to any known object of that class in the engine
        for track_id, state in self.engine.risk_map().items():
            if state["class_name"] == class_name:
                return track_id
        return None

    # -- reporting ---------------------------------------------------------
    def objects(self) -> list:
        risk_map = self.engine.risk_map()
        return sorted(risk_map.values(), key=lambda s: s["risk_score"], reverse=True)

    def snapshot(self) -> dict:
        risk_map = self.engine.risk_map()
        return {
            "source_kind": self.source_kind,
            # `source` mirrors source_kind under the dashboard-contract field name;
            # "mock" means the YOLO detector was NOT invoked this run (replay input).
            "source": self.source_kind,
            # The 8-class detector this pipeline is configured for (basename only).
            "model": os.path.basename(YOLO_MODEL_PATH),
            # Honesty flag: no physical kitchen/camera validation has happened.
            "physical_verification": False,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "tracked_objects": [
                {"track_id": t.track_id, "class_name": t.class_name,
                 "bbox_xyxy": [round(v, 1) for v in t.bbox]}
                for t in self.tracks
            ],
            "objects": self.objects(),
            "explanations": build_explanations(risk_map),
            "alerts": build_alerts(risk_map),
            "active_contacts": self.contacts.active_contacts(),
            "timeline": list(self.timeline),
        }


if __name__ == "__main__":
    from vision.mock_detection_source import MockDetectionSource

    pipe = RiskPipeline(source_kind="mock")
    src = MockDetectionSource("flagship_chain")
    for fd in src.frames():
        pipe.process_frame(fd)
    pipe.finish()
    snap = pipe.snapshot()
    print("Objects (risk desc):")
    for obj in snap["objects"]:
        print(f"  {obj['class_name']:<16} {obj['risk_class']:<7} "
              f"score={obj['risk_score']:.3f} depth={obj['propagation_depth']} chain={obj['risk_chain']}")
    print("Explanations:")
    for exp in snap["explanations"]:
        print(f"  {exp['object']}: {exp['chain_text'].encode('ascii','replace').decode()} "
              f"({exp['risk_class']} {exp['risk_score']:.3f})")
