"""Deterministic mock detection source (Phase 3).

Because the final 8-class YOLO `best.pt` is still training, this module stands in
for the detector: it emits realistic, tracked-object *detections* frame by frame
for a set of scripted kitchen scenarios. It behaves like a detection source, NOT
like a hidden answer key -- it never emits risk; risk is computed downstream by
the real contact tracker + Random Forest.

Design that keeps the reused IoU tracker's IDs stable while still producing
genuine *geometric* contacts over time: every object rests at a fixed "home" box
with a gap wider than the contact threshold, and a contact is created by sliding
ONE neighbour a short, smooth distance toward another until their boxes overlap
slightly, holding for several frames (so the contact tracker's persistence fires),
then sliding back. Per-frame displacement stays small, so each object's box
overlaps its own previous-frame box far more than any other -- the greedy IoU
tracker never swaps identities.

Cleaning is not something vision can detect reliably, so the cleaning scenario
emits a scripted *control event* (not a detection) that the demo controller / API
applies via RiskEngine.mark_cleaned -- matching Phase 10.

Same source interface the real YOLO adapter (vision/yolo_detection_source.py)
implements: `frames()` yields FrameData(frame_index, timestamp, detections,
control_events). Deterministic under a fixed seed (default 42).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.contracts import Detection  # noqa: E402

BBox = Tuple[float, float, float, float]

# Geometry knobs (pixels). Chosen so resting objects are clearly apart and a
# slide produces a small, unambiguous overlap.
_BOX_W = 80.0
_BOX_H = 120.0
_ROW_Y = 140.0
_ROW_X0 = 100.0
_REST_GAP = 70.0        # edge gap between neighbours at rest (> contact threshold 40)
# How much a mover overlaps its target at closest approach. ~45px on an 80px box
# gives a firm ~0.4 IoU contact (a realistic sustained touch, not a light graze),
# while keeping the movers' centres well separated so the greedy IoU tracker
# never swaps identities (verified by the stable-id integration test).
_OVERLAP_PX = 45.0

# Beat timing (frames). A ~1.8s dwell at 10 fps depicts a sustained contact.
_APPROACH = 6
_DWELL = 18
_RETREAT = 6
_REST_BETWEEN = 8       # frames apart between beats (lets a contact fully end)
_LEAD_IN = 8            # opening frames with everything at home ("source detected")
_TAIL = 8


@dataclass
class FrameData:
    frame_index: int
    timestamp: float
    detections: List[Detection]
    control_events: List[dict] = field(default_factory=list)


@dataclass
class _Beat:
    mover: str
    target: str
    start: int          # first frame of the beat


@dataclass
class _ScenarioDef:
    homes: Dict[str, BBox]
    beats: List[_Beat]
    controls: Dict[int, List[dict]]   # frame_index -> control events
    total_frames: int


def _row(names: List[str]) -> Dict[str, BBox]:
    """Lay objects out left-to-right in one row with a rest gap wider than the
    contact threshold, so nothing is 'in contact' at rest."""
    homes: Dict[str, BBox] = {}
    x = _ROW_X0
    for name in names:
        homes[name] = (x, _ROW_Y, x + _BOX_W, _ROW_Y + _BOX_H)
        x += _BOX_W + _REST_GAP
    return homes


def _beat_len() -> int:
    return _APPROACH + _DWELL + _RETREAT


def _mover_offset(homes: Dict[str, BBox], mover: str, target: str, frac: float) -> Tuple[float, float]:
    """Displacement of `mover` from its home at approach fraction `frac` in [0,1]
    (0 == home, 1 == overlapping the target by _OVERLAP_PX). Objects share a row,
    so motion is purely horizontal and the overlap is exact."""
    mx1, _, mx2, _ = homes[mover]
    tx1, _, tx2, _ = homes[target]
    mover_cx = (mx1 + mx2) / 2.0
    target_cx = (tx1 + tx2) / 2.0
    direction = 1.0 if target_cx > mover_cx else -1.0
    # edge gap at rest along x:
    gap = (tx1 - mx2) if direction > 0 else (mx1 - tx2)
    slide = (gap + _OVERLAP_PX) * frac
    return (direction * slide, 0.0)


def _build_sequential(homes, ordered_beats, controls_by_class=None):
    """Place beats back-to-back with rest between them, resolving class-named
    control events (e.g. cleaning) to their scheduled frame."""
    controls_by_class = controls_by_class or {}
    beats: List[_Beat] = []
    controls: Dict[int, List[dict]] = {}
    frame = _LEAD_IN
    for spec in ordered_beats:
        if spec.get("type") == "control":
            controls.setdefault(frame, []).append(
                {"type": spec["control"], "class": spec["class"]})
            frame += _REST_BETWEEN
            continue
        beats.append(_Beat(mover=spec["mover"], target=spec["target"], start=frame))
        frame += _beat_len() + _REST_BETWEEN
    total = frame + _TAIL
    return _ScenarioDef(homes=homes, beats=beats, controls=controls, total_frames=total)


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------
def _flagship_chain() -> _ScenarioDef:
    homes = _row(["nut_butter_jar", "cutlery", "bread", "plate"])
    beats = [
        {"mover": "cutlery", "target": "nut_butter_jar"},
        {"mover": "bread", "target": "cutlery"},
        {"mover": "plate", "target": "bread"},
    ]
    return _build_sequential(homes, beats)


def _direct_source_contact() -> _ScenarioDef:
    homes = _row(["whole_nuts", "hand", "bowl"])
    beats = [
        {"mover": "hand", "target": "whole_nuts"},
        {"mover": "bowl", "target": "hand"},
    ]
    return _build_sequential(homes, beats)


def _cleaning_interrupts_chain() -> _ScenarioDef:
    homes = _row(["nut_butter_jar", "cutlery", "bread"])
    beats = [
        {"mover": "cutlery", "target": "nut_butter_jar"},
        {"type": "control", "control": "cleaning", "class": "cutlery"},
        {"mover": "bread", "target": "cutlery"},
    ]
    return _build_sequential(homes, beats)


def _safe_unrelated_contacts() -> _ScenarioDef:
    # No allergen source anywhere in the scene -> risk should stay LOW.
    homes = _row(["plate", "bowl"])
    beats = [{"mover": "bowl", "target": "plate"}]
    return _build_sequential(homes, beats)


def _repeated_contact() -> _ScenarioDef:
    homes = _row(["nut_butter_jar", "cutlery"])
    beats = [
        {"mover": "cutlery", "target": "nut_butter_jar"},
        {"mover": "cutlery", "target": "nut_butter_jar"},
        {"mover": "cutlery", "target": "nut_butter_jar"},
    ]
    return _build_sequential(homes, beats)


SCENARIOS = {
    "flagship_chain": _flagship_chain,
    "direct_source_contact": _direct_source_contact,
    "cleaning_interrupts_chain": _cleaning_interrupts_chain,
    "safe_unrelated_contacts": _safe_unrelated_contacts,
    "repeated_contact": _repeated_contact,
}


def list_scenarios() -> List[str]:
    return list(SCENARIOS)


class MockDetectionSource:
    """Emits FrameData for a scripted scenario. Same `frames()` interface the
    YOLO adapter implements, so switching sources needs no pipeline change."""

    def __init__(self, scenario: str = "flagship_chain", fps: float = 10.0, seed: int = 42):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {list_scenarios()}")
        self.scenario = scenario
        self.fps = float(fps)
        self.seed = int(seed)
        self._def = SCENARIOS[scenario]()
        self._rng = np.random.default_rng(seed)

    @property
    def source_kind(self) -> str:
        return "mock"

    @property
    def total_frames(self) -> int:
        return self._def.total_frames

    def _active_offset(self, frame_index: int, name: str) -> Tuple[float, float]:
        length = _beat_len()
        for beat in self._def.beats:
            if beat.mover != name:
                continue
            t = frame_index - beat.start
            if 0 <= t < length:
                if t < _APPROACH:
                    frac = (t + 1) / _APPROACH
                elif t < _APPROACH + _DWELL:
                    frac = 1.0
                else:
                    frac = 1.0 - (t - _APPROACH - _DWELL + 1) / _RETREAT
                frac = float(np.clip(frac, 0.0, 1.0))
                return _mover_offset(self._def.homes, beat.mover, beat.target, frac)
        return (0.0, 0.0)

    def _frame(self, frame_index: int) -> FrameData:
        timestamp = frame_index / self.fps
        detections: List[Detection] = []
        for name, home in self._def.homes.items():
            dx, dy = self._active_offset(frame_index, name)
            jx = float(self._rng.normal(0.0, 0.4))
            jy = float(self._rng.normal(0.0, 0.4))
            x1, y1, x2, y2 = home
            bbox = (x1 + dx + jx, y1 + dy + jy, x2 + dx + jx, y2 + dy + jy)
            conf = float(np.clip(0.9 + self._rng.normal(0.0, 0.02), 0.5, 0.99))
            detections.append(
                Detection.from_class_name(name, conf, bbox, frame_index, timestamp))
        controls = list(self._def.controls.get(frame_index, []))
        return FrameData(frame_index=frame_index, timestamp=timestamp,
                         detections=detections, control_events=controls)

    def frames(self):
        """Yield one FrameData per frame for the whole scenario (deterministic)."""
        # Reset RNG so repeated iterations are identical.
        self._rng = np.random.default_rng(self.seed)
        for frame_index in range(self._def.total_frames):
            yield self._frame(frame_index)


if __name__ == "__main__":
    src = MockDetectionSource("flagship_chain")
    print(f"scenario=flagship_chain frames={src.total_frames} fps={src.fps}")
    for fd in src.frames():
        tags = ",".join(f"{d.class_name}" for d in fd.detections)
        ctl = f"  CONTROL={fd.control_events}" if fd.control_events else ""
        if fd.frame_index % 10 == 0 or ctl:
            print(f"  f{fd.frame_index:>3} t={fd.timestamp:5.2f}  [{tags}]{ctl}")
