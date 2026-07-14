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


class ContaminationTracker:
    """Sticky, class-keyed cross-contamination memory for one stream session."""

    def __init__(self, source_classes=None):
        # Carriers-from-the-start: the nut allergen sources ("the peanut butter").
        self.source_classes = set(
            source_classes if source_classes is not None else ALLERGEN_SOURCE_CLASSES.keys()
        )
        self.infected = set()        # class names that have been contaminated (sticky)
        self.sources_seen = set()    # source classes actually observed on camera
        self.notifications = []      # chronological log, one entry per NEW infection
        self.frame_index = 0

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

    # -- update ------------------------------------------------------------
    def observe(self, pairs, *, frame_index=0, timestamp=0.0, present_classes=None):
        """Apply the infection rules for one frame.

        `pairs` is an iterable of (classA, classB) that are TOUCHING this frame.
        Propagation is run to a fixed point so a fresh chain that lands in a
        single frame (source -> A -> B) fully resolves. Appends one notification
        per newly-infected class and returns the list of newly-infected names.
        """
        self.frame_index = frame_index
        for c in (present_classes or ()):
            if c in self.source_classes:
                self.sources_seen.add(c)

        edges = [(a, b) for a, b in pairs if a != b]  # ignore same-class double-detections
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
                    self.notifications.append({
                        "item": other,
                        "via": "peanut butter" if from_source else carrier,
                        "via_kind": "source" if from_source else "item",
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "message": (f"{other} contaminated by the peanut butter"
                                    if from_source else
                                    f"{other} contaminated by contact with {carrier}"),
                    })
                    changed = True
        return newly

    # -- serialization -----------------------------------------------------
    def state(self) -> dict:
        """Flat snapshot for GET /api/camera/contamination (and the DOM poller)."""
        return {
            "infected": sorted(self.infected),
            "sources": sorted(self.source_classes),
            "sources_seen": sorted(self.sources_seen),
            "notifications": list(self.notifications),
            "count": len(self.infected),
            "frame_index": self.frame_index,
        }
