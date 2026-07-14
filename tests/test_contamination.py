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
        infections = [n for n in t.notifications if n["kind"] == "infection"]
        self.assertEqual(len(infections), 1)
        note = infections[0]
        self.assertEqual(note["item"], "cutlery")
        self.assertEqual(note["via_kind"], "source")
        self.assertIn("peanut butter", note["message"])

    def test_allergen_detection_emits_one_notification(self):
        t = ContaminationTracker()
        # peanut butter appears in view (nothing touching yet)
        t.observe([], frame_index=1, present_classes=[JAR, "plate"])
        allergens = [n for n in t.notifications if n["kind"] == "allergen"]
        self.assertEqual(len(allergens), 1)
        self.assertEqual(allergens[0]["item"], JAR)
        self.assertIn("Allergen detected", allergens[0]["message"])
        self.assertEqual(t.state()["sources_seen"], [JAR])

    def test_allergen_notification_not_duplicated(self):
        t = ContaminationTracker()
        t.observe([], frame_index=1, present_classes=[JAR])
        t.observe([], frame_index=2, present_classes=[JAR])   # still in view next frame
        allergens = [n for n in t.notifications if n["kind"] == "allergen"]
        self.assertEqual(len(allergens), 1)

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


class ContaminationRisk(unittest.TestCase):
    JARBOX = (0, 0, 100, 100)   # source bbox: top y=0, bottom y=100

    def test_touch_top_of_peanut_butter_is_high_risk(self):
        t = ContaminationTracker()
        # cutlery contacting the TOP of the jar
        t.observe([(JAR, "cutlery")], boxes={JAR: self.JARBOX, "cutlery": (10, -40, 90, 0)})
        self.assertAlmostEqual(t.risk_of("cutlery"), 0.9, places=3)

    def test_touch_bottom_of_peanut_butter_is_low_risk(self):
        t = ContaminationTracker()
        # cutlery contacting the BOTTOM of the jar
        t.observe([(JAR, "cutlery")], boxes={JAR: self.JARBOX, "cutlery": (10, 100, 90, 140)})
        self.assertAlmostEqual(t.risk_of("cutlery"), 0.4, places=3)

    def test_middle_contact_is_between(self):
        t = ContaminationTracker()
        t.observe([(JAR, "cutlery")], boxes={JAR: self.JARBOX, "cutlery": (10, 40, 90, 60)})
        self.assertAlmostEqual(t.risk_of("cutlery"), 0.65, places=3)   # midpoint

    def test_risk_decays_along_the_spread_chain(self):
        t = ContaminationTracker()
        t.observe([(JAR, "cutlery")], boxes={JAR: self.JARBOX, "cutlery": (10, -40, 90, 0)})  # 0.9
        t.observe([("cutlery", "bread")])     # spread: 0.9 * 0.6
        t.observe([("bread", "plate")])       # spread: 0.54 * 0.6
        self.assertAlmostEqual(t.risk_of("cutlery"), 0.9, places=3)
        self.assertAlmostEqual(t.risk_of("bread"), 0.54, places=3)
        self.assertAlmostEqual(t.risk_of("plate"), 0.324, places=3)

    def test_risk_without_geometry_defaults_to_worst_case(self):
        t = ContaminationTracker()
        t.observe([(JAR, "cutlery")])         # no boxes -> conservative default
        self.assertAlmostEqual(t.risk_of("cutlery"), 0.9, places=3)

    def test_clean_item_has_zero_risk_and_risk_in_state(self):
        t = ContaminationTracker()
        t.observe([(JAR, "cutlery")])
        self.assertEqual(t.risk_of("plate"), 0.0)
        self.assertIn("cutlery", t.state()["risk"])
        self.assertNotIn("plate", t.state()["risk"])

    def test_infection_notification_carries_the_risk(self):
        t = ContaminationTracker()
        t.observe([(JAR, "cutlery")], boxes={JAR: self.JARBOX, "cutlery": (10, -40, 90, 0)})
        note = [n for n in t.notifications if n["kind"] == "infection"][0]
        self.assertAlmostEqual(note["risk"], 0.9, places=3)
        self.assertIn("0.90", note["message"])


if __name__ == "__main__":
    unittest.main()
