"""Contamination memory for the live camera stream.

A deliberately simple, deterministic layer that sits ON TOP of the per-frame
box-overlap contact cue (pipeline.live_yolo_runner.draw_collision_overlay). It
answers a question the graded RF risk model does not: "which items have been
cross-contaminated with the nut allergen, and how did it spread?"

Rules (binary + STICKY, keyed by CLASS NAME because the operator uses only one
of each object -- so once "cutlery" is infected it is remembered as infected
even after it leaves and re-enters the frame with a new track id):

  * The allergen SOURCE classes (nut_butter_jar / whole_nuts -- "the peanut
    butter") are carriers from the start.
  * Any item that TOUCHES a carrier becomes INFECTED and stays infected for the
    rest of the stream session.
  * An infected item is itself a carrier: whatever it later touches also becomes
    infected (transitive propagation).

"Touch" is bounding-box overlap -- the same signal the CONTACT overlay uses; it
is a proxy for contact, not a biochemical measurement (see camera.html limits).

The memory is per-instance: the live stream builds a FRESH tracker every time it
(re)starts, so "restart stream" reloads the checker with an empty slate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import ALLERGEN_SOURCE_CLASSES

# --- Graded contact-risk model (deterministic; tunable) --------------------
# Touching the SOURCE (peanut butter) is riskier at the TOP (near the opening /
# the spread) than at the BOTTOM (the base of the jar). The initial risk is
# interpolated by WHERE, vertically, the contact lands within the source's box.
SOURCE_TOP_RISK = 0.9          # contact at the very top of the peanut butter
SOURCE_BOTTOM_RISK = 0.4       # contact at the very bottom
SOURCE_DEFAULT_RISK = SOURCE_TOP_RISK   # worst-case fallback when geometry is unknown
# Each further hop of spread multiplies the risk down (contamination dilutes as
# it travels object -> object): 0.9 -> 0.54 -> 0.32 -> ...
SPREAD_DECAY = 0.6


class ContaminationTracker:
    """Sticky, class-keyed cross-contamination memory for one stream session."""

    def __init__(self, source_classes=None):
        # Carriers-from-the-start: the nut allergen sources ("the peanut butter").
        self.source_classes = set(
            source_classes if source_classes is not None else ALLERGEN_SOURCE_CLASSES.keys()
        )
        self.infected = set()        # class names that have been contaminated (sticky)
        self.risk = {}               # class name -> graded contamination risk (0..1)
        self.sources_seen = set()    # source classes actually observed on camera
        self.notifications = []      # chronological log: allergen-detected + infection events
        self.frame_index = 0
        self.new_allergens = []      # sources first seen THIS frame (for the on-frame flash)
        self.new_infections = []     # items infected THIS frame (for the on-frame flash)

    # -- queries -----------------------------------------------------------
    def is_carrier(self, class_name: str) -> bool:
        """A carrier can pass contamination on: the source, or an infected item."""
        return class_name in self.source_classes or class_name in self.infected

    def status(self, class_name: str) -> str:
        """'source' | 'infected' | 'clean' -- used to tag boxes on the frame."""
        if class_name in self.source_classes:
            return "source"
        if class_name in self.infected:
            return "infected"
        return "clean"

    def risk_of(self, class_name: str) -> float:
        """Current graded contamination risk (0..1) for a class; 0.0 if clean."""
        return self.risk.get(class_name, 0.0)

    def _source_risk(self, source_box, item_box) -> float:
        """Risk from touching the SOURCE, graded by WHERE the contact lands on it:
        the TOP of the peanut butter -> SOURCE_TOP_RISK, the BOTTOM ->
        SOURCE_BOTTOM_RISK, linearly interpolated by the vertical position of the
        contact within the source's bounding box. Falls back to the worst case
        when contact geometry is unavailable (e.g. no boxes were passed)."""
        if not source_box:
            return SOURCE_DEFAULT_RISK
        sx1, sy1, sx2, sy2 = source_box
        if item_box:
            iy1, iy2 = max(sy1, item_box[1]), min(sy2, item_box[3])
            contact_y = (iy1 + iy2) / 2.0 if iy2 > iy1 else (item_box[1] + item_box[3]) / 2.0
        else:
            contact_y = (sy1 + sy2) / 2.0
        height = max(1.0, float(sy2 - sy1))
        t = min(1.0, max(0.0, (contact_y - sy1) / height))   # 0 at top, 1 at bottom
        return round(SOURCE_TOP_RISK - t * (SOURCE_TOP_RISK - SOURCE_BOTTOM_RISK), 3)

    # -- update ------------------------------------------------------------
    def observe(self, pairs, *, frame_index=0, timestamp=0.0, present_classes=None, boxes=None):
        """Apply the detection + infection rules for one frame.

        `pairs` is an iterable of (classA, classB) that are TOUCHING this frame.
        `boxes` (optional) maps class name -> (x1, y1, x2, y2); when given, a
        contact with the source is risk-graded by WHERE on the source it lands
        (top of the peanut butter = high, bottom = low), and each further hop of
        spread decays the risk. Emits two kinds of notification:
          * "allergen"  -- once, the first time a source (peanut butter) is seen.
          * "infection" -- once per newly-infected item (carries its risk score);
            propagation runs to a fixed point so a chain landing in a single
            frame fully resolves.
        Returns the list of newly-infected class names (used for the on-frame flash).
        """
        boxes = boxes or {}
        self.frame_index = frame_index
        edges = [(a, b) for a, b in pairs if a != b]  # ignore same-class double-detections

        # Everything visible this frame: explicit detections + anything touching.
        present = set(present_classes or ())
        for a, b in edges:
            present.add(a)
            present.add(b)

        # (1) Allergen-detected notification -- one per source, the first time it appears.
        new_allergens = []
        for c in sorted(present):
            if c in self.source_classes and c not in self.sources_seen:
                self.sources_seen.add(c)
                new_allergens.append(c)
                self.notifications.append({
                    "kind": "allergen",
                    "item": c,
                    "via": None,
                    "via_kind": "detection",
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "message": f"Allergen detected: {c} (the peanut butter)",
                })
        self.new_allergens = new_allergens

        # (2) Infection notifications -- propagate carriers to a fixed point.
        newly = []
        changed = True
        while changed:
            changed = False
            for a, b in edges:
                for carrier, other in ((a, b), (b, a)):
                    if not self.is_carrier(carrier):
                        continue
                    if other in self.source_classes or other in self.infected:
                        continue
                    self.infected.add(other)
                    newly.append(other)
                    from_source = carrier in self.source_classes
                    if from_source:
                        risk = self._source_risk(boxes.get(carrier), boxes.get(other))
                    else:
                        # Spread from an already-infected carrier: decay its risk.
                        risk = round(self.risk.get(carrier, SOURCE_DEFAULT_RISK) * SPREAD_DECAY, 3)
                    self.risk[other] = max(self.risk.get(other, 0.0), risk)
                    self.notifications.append({
                        "kind": "infection",
                        "item": other,
                        "via": "peanut butter" if from_source else carrier,
                        "via_kind": "source" if from_source else "item",
                        "risk": risk,
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "message": (f"{other} contaminated by the peanut butter (risk {risk:.2f})"
                                    if from_source else
                                    f"{other} contaminated by contact with {carrier} (risk {risk:.2f})"),
                    })
                    changed = True
        self.new_infections = newly
        return newly

    # -- serialization -----------------------------------------------------
    def state(self) -> dict:
        """Flat snapshot for GET /api/camera/contamination (and the DOM poller)."""
        return {
            "infected": sorted(self.infected),
            "risk": {k: round(v, 3) for k, v in self.risk.items()},
            "sources": sorted(self.source_classes),
            "sources_seen": sorted(self.sources_seen),
            "notifications": list(self.notifications),
            "count": len(self.infected),
            "frame_index": self.frame_index,
        }
