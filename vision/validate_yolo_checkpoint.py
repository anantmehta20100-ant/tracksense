"""Validate that a YOLO checkpoint is the correct TrackSense 8-class detector.

Confirms `model/checkpoints/tracksense_8class_best.pt` (or any given weights) is
the right model BEFORE the live pipeline uses it. It does NOT retrain, modify
weights, or run inference -- it only reads `model.names` and checks the class
schema.

Single source of truth: the expected schema comes from
`ml/class_schema.training_names()` (exposed as
`config.runtime_config.EXPECTED_YOLO_CLASS_NAMES`), and the authoritative
exact-match check is `vision/yolo_detection_source.validate_class_names()`. This
utility adds explicit, individually-reported sub-checks (exactly 8 classes, no
`counter`, `bread == 7`, not a single-class model) for a clearer pass/fail report.

CLI:
    python vision/validate_yolo_checkpoint.py --model model/checkpoints/tracksense_8class_best.pt

Exit codes: 0 = valid, 1 = schema invalid, 2 = checkpoint file missing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import EXPECTED_YOLO_CLASS_NAMES, YOLO_MODEL_PATH  # noqa: E402
from vision.yolo_detection_source import (  # noqa: E402
    ModelSchemaMismatch,
    _normalize_names,
    validate_class_names,
)

# Normalized {int: str}, straight from the single source of truth.
EXPECTED_SCHEMA: Dict[int, str] = {int(k): v for k, v in EXPECTED_YOLO_CLASS_NAMES.items()}

# The old single-class checkpoint that must never be accepted as the detector.
LEGACY_CHECKPOINT = os.path.join("model", "checkpoints", "best.pt")


def check_schema(model_names) -> Tuple[bool, Dict[int, str], List[Tuple[str, bool, str]]]:
    """Run every schema check against a model's class names.

    Returns (ok, normalized_names, checks) where checks is a list of
    (label, passed, detail). Pure and GPU-free -- takes names, not a model.
    """
    names = _normalize_names(model_names)
    values = list(names.values())
    checks: List[Tuple[str, bool, str]] = []

    checks.append((
        "exactly 8 classes (not 9)", len(names) == 8, f"found {len(names)} classes",
    ))
    checks.append((
        "`counter` absent (not part of the trained detector)",
        "counter" not in values, "counter present" if "counter" in values else "ok",
    ))
    checks.append((
        "bread is local id 7", names.get(7) == "bread", f"id 7 = {names.get(7)!r}",
    ))
    checks.append((
        "not a single-class model (e.g. old cutlery best.pt)",
        len(names) > 1, f"{len(names)} class(es): {values}" if len(names) <= 1 else "ok",
    ))

    try:
        validate_class_names(names)
        exact_ok, exact_detail = True, "exact match with the 8-class schema"
    except ModelSchemaMismatch:
        exact_ok, exact_detail = False, "does not match the expected 8-class schema"
    checks.append(("exact schema match (ids + names)", exact_ok, exact_detail))

    ok = all(passed for _, passed, _ in checks)
    return ok, names, checks


def load_model_names(model_path: str):
    """Load the checkpoint with ultralytics and return model.names. Raises
    FileNotFoundError if the checkpoint is missing (never falls back to another
    file)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    from ultralytics import YOLO  # lazy: only needed when a real checkpoint exists

    return YOLO(model_path).names


def _format_schema(names: Dict[int, str]) -> str:
    return "\n".join(f"{i} {names[i]}" for i in sorted(names))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a YOLO checkpoint's class schema.")
    parser.add_argument("--model", default=YOLO_MODEL_PATH,
                        help="Path to the YOLO checkpoint (.pt). Default: %(default)s")
    args = parser.parse_args(argv)

    print("Expected TrackSense 8-class schema:")
    print(_format_schema(EXPECTED_SCHEMA))
    print(f"\nChecking checkpoint: {args.model}")

    try:
        model_names = load_model_names(args.model)
    except FileNotFoundError:
        print("\nCHECKPOINT MISSING.")
        print(f"  No file at '{args.model}'.")
        print("  Place the downloaded Kaggle 8-class checkpoint there "
              "(or pass --model /path/to/weights.pt).")
        if os.path.exists(LEGACY_CHECKPOINT) and os.path.abspath(LEGACY_CHECKPOINT) != os.path.abspath(args.model):
            print(f"  NOTE: '{LEGACY_CHECKPOINT}' exists but must NOT be used -- it is the old "
                  "checkpoint. Do not rename it to the 8-class path; validate the real model instead.")
        return 2
    except Exception as exc:  # noqa: BLE001 - surface any ultralytics load error clearly
        print(f"\nFAILED TO LOAD checkpoint with ultralytics: {type(exc).__name__}: {exc}")
        return 1

    ok, names, checks = check_schema(model_names)

    print("\nSchema checks:")
    for label, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {label}  ({detail})")

    print(f"\nDetected model classes: {names}")

    if ok:
        print("\nYOLO checkpoint schema valid:")
        print(_format_schema(names))
        return 0

    print("\nYOLO checkpoint schema INVALID -- refusing to use this checkpoint for the live pipeline.")
    print("This is often the old single-class cutlery model, or a model trained with a different "
          "class order. Do not use it. Supply the correct 8-class detector.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
