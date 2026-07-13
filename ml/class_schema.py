"""Explicit, centralized, testable class-schema mapping for TrackSense.

DECISION: train a temporary 8-class YOLO model that EXCLUDES `counter`, because
no counter dataset exists yet and no counter data will be fabricated.

The canonical 9-class project schema (config/allergens.py) is UNCHANGED:

    0 nut_butter_jar   3 cutlery          6 bowl
    1 whole_nuts       4 chopping_board   7 counter   <- canonical, zero samples
    2 hand             5 plate            8 bread

The temporary YOLO training model uses 8 CONTIGUOUS local ids (0..7). Every
populated class keeps its canonical number except bread, which is canonical 8
but model-local 7 (it slides down into the hole left by counter):

    local 0 nut_butter_jar   local 4 chopping_board
    local 1 whole_nuts       local 5 plate
    local 2 hand             local 6 bowl
    local 3 cutlery          local 7 bread   <- canonical 8

Rules enforced here:
  - canonical ids are never redefined; canonical 7 stays `counter` forever.
  - counter has NO model-local id (it is absent from the trained model).
  - bread is never silently remapped: the 8 -> 7 move lives only in this file
    and is applied explicitly by the dataset builders.
  - runtime code that consumes model output MUST call model_to_canonical() to
    get back to canonical TrackSense ids before touching config/allergens.py,
    the tracker, contact_detector, risk_state, consumption or the dashboard.

Run `python ml/class_schema.py` to execute the self-test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASSES, OBJECT_CLASS_TO_ID, OBJECT_ID_TO_CLASS

# Canonical classes that currently have zero labelled samples and are therefore
# excluded from the temporary training model. Never fabricate data for these.
EXCLUDED_FROM_TRAINING = ("counter",)

# The active training schema for this project phase.
TRAINING_MODE = "contiguous8"

# ---------------------------------------------------------------------------
# The explicit mapping (authoritative; do not duplicate these dicts elsewhere).
# ---------------------------------------------------------------------------
MODEL_LOCAL_TO_CANONICAL = {
    0: 0,  # nut_butter_jar
    1: 1,  # whole_nuts
    2: 2,  # hand
    3: 3,  # cutlery
    4: 4,  # chopping_board
    5: 5,  # plate
    6: 6,  # bowl
    7: 8,  # bread   (canonical 8 -> model-local 7; counter's slot 7 is skipped)
}

CANONICAL_TO_MODEL_LOCAL = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    8: 7,  # bread
}

# Model-local id -> class name, for the 8-class training YAML.
MODEL_LOCAL_NAMES = {
    local: OBJECT_ID_TO_CLASS[canonical] for local, canonical in MODEL_LOCAL_TO_CANONICAL.items()
}

NUM_TRAINING_CLASSES = len(MODEL_LOCAL_TO_CANONICAL)


def training_names() -> dict[int, str]:
    """Model-local id -> name, for the training data.yaml."""
    return dict(MODEL_LOCAL_NAMES)


def model_to_canonical(model_local_id: int) -> int:
    """Map a model output class id back to a canonical TrackSense id."""
    if model_local_id not in MODEL_LOCAL_TO_CANONICAL:
        raise KeyError(f"unknown model-local class id {model_local_id}")
    return MODEL_LOCAL_TO_CANONICAL[model_local_id]


def canonical_to_model(canonical_id: int) -> int:
    """Map a canonical TrackSense id to its model-local id.

    Raises for canonical classes excluded from training (e.g. counter=7).
    """
    if canonical_id not in CANONICAL_TO_MODEL_LOCAL:
        name = OBJECT_ID_TO_CLASS.get(canonical_id, "?")
        raise KeyError(
            f"canonical class {canonical_id} ({name}) has no model-local id "
            f"-- it is excluded from the {TRAINING_MODE} training schema"
        )
    return CANONICAL_TO_MODEL_LOCAL[canonical_id]


def is_trained(canonical_id: int) -> bool:
    return canonical_id in CANONICAL_TO_MODEL_LOCAL


def _selftest() -> None:
    excluded_ids = {OBJECT_CLASS_TO_ID[c] for c in EXCLUDED_FROM_TRAINING}

    # The two dicts must be exact inverses of one another.
    assert MODEL_LOCAL_TO_CANONICAL == {v: k for k, v in CANONICAL_TO_MODEL_LOCAL.items()}
    for local in MODEL_LOCAL_TO_CANONICAL:
        assert canonical_to_model(model_to_canonical(local)) == local, local
    for canonical in CANONICAL_TO_MODEL_LOCAL:
        assert model_to_canonical(canonical_to_model(canonical)) == canonical, canonical

    # Local ids are exactly 0..7, contiguous.
    assert sorted(MODEL_LOCAL_TO_CANONICAL) == list(range(8))
    assert NUM_TRAINING_CLASSES == 8

    # counter is canonical 7, excluded, and has no local id.
    assert OBJECT_CLASS_TO_ID["counter"] == 7
    assert excluded_ids == {7}
    assert 7 not in CANONICAL_TO_MODEL_LOCAL
    assert "counter" not in MODEL_LOCAL_NAMES.values()
    assert not is_trained(7)
    try:
        canonical_to_model(7)
        raise AssertionError("counter must not have a model-local id")
    except KeyError:
        pass

    # bread: canonical 8 -> local 7, and local 7 -> canonical 8.
    assert OBJECT_CLASS_TO_ID["bread"] == 8
    assert canonical_to_model(8) == 7
    assert model_to_canonical(7) == 8
    assert MODEL_LOCAL_NAMES[7] == "bread"

    # Every populated canonical class other than bread keeps its number.
    for canonical, local in CANONICAL_TO_MODEL_LOCAL.items():
        if OBJECT_ID_TO_CLASS[canonical] != "bread":
            assert canonical == local, (canonical, local)

    # Canonical project schema untouched: still 9 classes in the original order.
    assert len(OBJECT_CLASSES) == 9
    assert OBJECT_ID_TO_CLASS[7] == "counter"

    # Every trained local name resolves to the right canonical name.
    for local, name in MODEL_LOCAL_NAMES.items():
        assert OBJECT_ID_TO_CLASS[model_to_canonical(local)] == name

    print("class_schema self-test PASS")
    print(f"TRAINING_MODE       = {TRAINING_MODE}  ({NUM_TRAINING_CLASSES} classes)")
    print(f"excluded from model = {EXCLUDED_FROM_TRAINING} (canonical ids {sorted(excluded_ids)})")
    print("local -> canonical  =", MODEL_LOCAL_TO_CANONICAL)
    print("canonical -> local  =", CANONICAL_TO_MODEL_LOCAL)
    print("training names      =", MODEL_LOCAL_NAMES)


if __name__ == "__main__":
    _selftest()
