"""
Central registry for object classes, allergen types, and tunable thresholds.
Every other module imports from here instead of hardcoding these values.
"""

# ---------------------------------------------------------------------------
# Object classes (guaranteed nuts-only scope, see AGENTS.md "Object classes")
# ---------------------------------------------------------------------------
ALLERGEN_SOURCE_CLASSES = {
    "nut_butter_jar": "nut",
    "whole_nuts": "nut",
}

UTENSIL_CLASSES = ["hand", "cutlery", "chopping_board"]
SURFACE_CLASSES = ["plate", "bowl", "counter"]
FOOD_CLASSES = ["bread", "whole_nuts", "nut_butter_jar"]

OBJECT_CLASSES = [
    "nut_butter_jar",
    "whole_nuts",
    "hand",
    "cutlery",
    "chopping_board",
    "plate",
    "bowl",
    "counter",
    "bread",
]
OBJECT_CLASS_TO_ID = {class_name: index for index, class_name in enumerate(OBJECT_CLASSES)}
OBJECT_ID_TO_CLASS = {index: class_name for class_name, index in OBJECT_CLASS_TO_ID.items()}

# All allergen types the system currently understands. Only "nut" is in the
# guaranteed scope; "dairy" is the documented stretch goal (AGENTS.md).
ALLERGEN_TYPES = ["nut"]


def get_allergen_type(class_name: str):
    """Look up the allergen_type for a source object class, or None if the
    class is not an allergen source (e.g. cutlery or plate)."""
    return ALLERGEN_SOURCE_CLASSES.get(class_name)


# ---------------------------------------------------------------------------
# Contact detection thresholds (vision/contact_detector.py, vision/tracker.py)
# ---------------------------------------------------------------------------

# Max pixel distance (or bbox overlap gap) between two tracked objects to
# count as "in contact". Tunable: depends on camera resolution/distance from
# the counter. Default assumes a ~1080p webcam framing a kitchen counter.
CONTACT_PROXIMITY_THRESHOLD_PX = 40

# Number of consecutive frames two objects must stay within the proximity
# threshold before a contact event is emitted. Filters out momentary/noisy
# overlaps from detector jitter.
CONTACT_PERSISTENCE_FRAMES = 5

# IoU tracker: minimum IoU between a new detection and a previous track's
# bbox to consider them the same object.
TRACKER_IOU_MATCH_THRESHOLD = 0.3

# Number of consecutive frames a track can go unmatched before it is dropped.
TRACKER_MAX_MISSED_FRAMES = 10

# ---------------------------------------------------------------------------
# Consumption / exposure alert thresholds (pipeline/consumption.py)
# ---------------------------------------------------------------------------

# Pixel distance between a food object's bbox and the mouth region to count
# as "near the mouth".
MOUTH_PROXIMITY_THRESHOLD_PX = 60

# Consecutive frames a food object must stay near the mouth before it's
# logged as a consumption event.
CONSUMPTION_PERSISTENCE_FRAMES = 8

# If a consumption event's risk_at_time (for the user's stated allergen_type)
# is >= this value, trigger the exposure alert.
EXPOSURE_ALERT_RISK_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Risk propagation model shared constants (model/, pipeline/risk_state.py)
# ---------------------------------------------------------------------------

# Fixed per-hop decay rate used by both the synthetic data label rule and the
# RuleBasedDecayBaseline, e.g. 100% -> 70% -> 49% -> 34% (AGENTS.md example).
RULE_BASED_DECAY_RATE = 0.7

# Risk value an object is reset to (before noise) when a cleaning event
# occurs.
CLEANING_RISK_RESET_VALUE = 0.05
