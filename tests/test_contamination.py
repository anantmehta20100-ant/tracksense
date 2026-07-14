"""Unit tests for the live-stream contamination memory (no camera / no YOLO).

Exercises the rules the operator specified:
  * touching the peanut butter infects an item (+ a notification),
  * an infected item infects whatever it then touches (propagation),
  * infection is sticky and class-keyed (survives frames where nothing touches),
  * a fresh tracker is empty (= "restart stream reloads the checker").
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.contamination import ContaminationTracker

JAR = "nut_butter_jar"


class ContaminationRules(unittest.TestCase):
    def test_source_is_carrier_but_not_listed_infected(self):
        t = ContaminationTracker()
        self.assertTrue(t.is_carrier(JAR))
        self.assertEqual(t.status(JAR), "source")
        self.assertEqual(t.state()["infected"], [])

    def test_touching_peanut_butter_infects_and_notifies(self):
        t = ContaminationTracker()
        newly = t.observe([(JAR, "cutlery")], frame_index=1, present_classes=[JAR, "cutlery"])
        self.assertEqual(newly, ["cutlery"])
        self.assertEqual(t.status("cutlery"), "infected")
        self.assertEqual(len(t.notifications), 1)
        note = t.notifications[0]
        self.assertEqual(note["item"], "cutlery")
        self.assertEqual(note["via_kind"], "source")
        self.assertIn("peanut butter", note["message"])

    def test_infected_item_propagates_to_next_item(self):
        t = ContaminationTracker()
        t.observe([(JAR, "cutlery")], frame_index=1)          # cutlery infected
        newly = t.observe([("cutlery", "bread")], frame_index=2)  # bread via cutlery
        self.assertEqual(newly, ["bread"])
        self.assertEqual(t.status("bread"), "infected")
        self.assertEqual(t.notifications[-1]["via"], "cutlery")
        self.assertEqual(t.notifications[-1]["via_kind"], "item")

    def test_infection_is_sticky_across_frames(self):
        t = ContaminationTracker()
        t.observe([(JAR, "cutlery")], frame_index=1)
        for f in range(2, 6):                                  # frames with NO contact
            t.observe([], frame_index=f)
        self.assertEqual(t.status("cutlery"), "infected")      # still remembered

    def test_clean_item_that_never_touches_stays_clean(self):
        t = ContaminationTracker()
        t.observe([(JAR, "cutlery")], frame_index=1)
        self.assertEqual(t.status("plate"), "clean")
        self.assertNotIn("plate", t.state()["infected"])

    def test_no_duplicate_notification_for_same_item(self):
        t = ContaminationTracker()
        t.observe([(JAR, "cutlery")], frame_index=1)
        t.observe([(JAR, "cutlery")], frame_index=2)           # touching again
        cutlery_notes = [n for n in t.notifications if n["item"] == "cutlery"]
        self.assertEqual(len(cutlery_notes), 1)                # notified exactly once

    def test_single_frame_chain_resolves_to_fixed_point(self):
        t = ContaminationTracker()
        # jar<->cutlery and cutlery<->bread both present in ONE frame
        newly = t.observe([(JAR, "cutlery"), ("cutlery", "bread")], frame_index=1)
        self.assertEqual(set(newly), {"cutlery", "bread"})
        self.assertEqual(t.status("bread"), "infected")

    def test_same_class_self_pair_is_ignored(self):
        t = ContaminationTracker()
        t.observe([("cutlery", "cutlery")], frame_index=1)     # double-detection noise
        self.assertEqual(t.state()["infected"], [])

    def test_fresh_tracker_is_empty_restart_reloads(self):
        t = ContaminationTracker()
        t.observe([(JAR, "cutlery")], frame_index=1)
        self.assertTrue(t.state()["infected"])
        fresh = ContaminationTracker()                          # what "restart stream" builds
        self.assertEqual(fresh.state()["infected"], [])
        self.assertEqual(fresh.notifications, [])


if __name__ == "__main__":
    unittest.main()
