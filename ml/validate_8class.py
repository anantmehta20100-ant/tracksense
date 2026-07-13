"""Strict validation of the final 8-class training candidate.

Hard checks (any failure -> non-zero exit):
  - every image has a label; every label has an image
  - all class ids are model-local 0..7
  - NO canonical class 8 labels remain (bread must be local 7)
  - NO counter labels (canonical 7 is excluded entirely)
  - x_center, y_center in [0,1]; width, height in (0,1]
  - no NaN, no infinity, no zero-area boxes
  - no unreadable images, no corrupt/malformed labels
  - no filename collisions (across the whole dataset)
  - no cross-split exact (SHA-256) duplicates

Advisory (flagged to reports/box_sanity_review.csv, never auto-deleted):
  - extremely tiny boxes
  - nearly full-frame boxes
  - extreme aspect ratios

Also prints the per-class train/valid/test image + instance table and flags
statistically weak evaluation sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.class_schema import MODEL_LOCAL_NAMES, NUM_TRAINING_CLASSES, model_to_canonical

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REVIEW_PATH = Path("reports/box_sanity_review.csv")

FULL_FRAME_AREA = 0.98
TINY_AREA = 0.0005
EXTREME_AR = 20.0
MIN_RELIABLE_EVAL = 30
CRITICAL_EVAL = 10


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def class_of(name: str) -> str:
    return name.split("__", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict validation of the 8-class training candidate.")
    parser.add_argument("--root", default="data/training_8class_balanced")
    args = parser.parse_args()
    root = Path(args.root)

    print(f"Validating 8-class candidate: {root}")
    print("=" * 60)

    errors: list[str] = []
    suspicious: list[dict] = []
    per_class = defaultdict(lambda: {s: {"img": 0, "inst": 0} for s in SPLITS})
    by_sha: dict[str, set] = defaultdict(set)
    all_names: dict[str, str] = {}
    unreadable = 0
    id_hist = defaultdict(int)

    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            errors.append(f"{split}: missing images/ or labels/ directory")
            continue

        images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {p.stem: p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}

        for stem in sorted(set(images) - set(labels)):
            errors.append(f"{split}: orphan image (no label): {stem}")
        for stem in sorted(set(labels) - set(images)):
            errors.append(f"{split}: orphan label (no image): {stem}")

        for stem, image_path in images.items():
            if image_path.name in all_names:
                errors.append(f"filename collision: {image_path.name} in {split} and {all_names[image_path.name]}")
            all_names[image_path.name] = split

            try:
                with Image.open(image_path) as im:
                    im.verify()
            except Exception as exc:
                errors.append(f"{split}: unreadable image {image_path.name}: {exc}")
                unreadable += 1
                continue

            by_sha[sha256_of(image_path)].add(split)
            cls = class_of(image_path.name)
            per_class[cls][split]["img"] += 1

            label_path = labels.get(stem)
            if not label_path:
                continue
            lines = [l for l in label_path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
            if not lines:
                errors.append(f"{split}: empty label {label_path.name}")
            for line_no, raw in enumerate(lines, 1):
                parts = raw.split()
                if len(parts) < 5:
                    errors.append(f"{label_path.name}:{line_no}: fewer than 5 fields")
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    errors.append(f"{label_path.name}:{line_no}: non-numeric field")
                    continue
                if any(math.isnan(v) or math.isinf(v) for v in (cx, cy, w, h)):
                    errors.append(f"{label_path.name}:{line_no}: NaN/inf coordinate")
                    continue
                if not (0 <= cid < NUM_TRAINING_CLASSES):
                    errors.append(f"{label_path.name}:{line_no}: class id {cid} outside model-local 0..7")
                    continue
                id_hist[cid] += 1
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    errors.append(f"{label_path.name}:{line_no}: center out of [0,1]")
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    errors.append(f"{label_path.name}:{line_no}: w/h not in (0,1]")
                    continue
                if w * h <= 0:
                    errors.append(f"{label_path.name}:{line_no}: zero-area box")
                    continue
                per_class[cls][split]["inst"] += 1

                area = w * h
                aspect = max(w / h, h / w)
                flags = []
                if area >= FULL_FRAME_AREA:
                    flags.append("near-full-frame")
                if area <= TINY_AREA:
                    flags.append("tiny")
                if aspect >= EXTREME_AR:
                    flags.append("extreme-aspect")
                if flags:
                    suspicious.append({
                        "split": split, "image": image_path.name, "class": cls,
                        "model_local_id": cid, "canonical_id": model_to_canonical(cid),
                        "line": line_no, "cx": cx, "cy": cy, "w": w, "h": h,
                        "area": round(area, 6), "aspect": round(aspect, 2), "flags": "|".join(flags),
                    })

    # Schema-specific hard assertions.
    if 8 in id_hist:
        errors.append("canonical class 8 (bread) label found in training labels -- must be local id 7")
    if any(cid >= NUM_TRAINING_CLASSES for cid in id_hist):
        errors.append("class id >= 8 present")
    if "counter" in per_class:
        errors.append("counter images present in the 8-class candidate")

    cross = [s for s in by_sha.values() if len(s) > 1]
    if cross:
        errors.append(f"{len(cross)} cross-split exact-duplicate group(s)")

    # ---- report ----
    print("Per-class counts (images | instances)")
    print(f"{'loc':>3} {'canon':>5} {'class':16} {'train':>13} {'valid':>13} {'test':>13}")
    print("-" * 72)
    weak, critical = [], []
    for local in range(NUM_TRAINING_CLASSES):
        cls = MODEL_LOCAL_NAMES[local]
        canon = model_to_canonical(local)
        d = per_class.get(cls, {s: {"img": 0, "inst": 0} for s in SPLITS})
        print(f"{local:>3} {canon:>5} {cls:16} "
              f"{d['train']['img']:>5}|{d['train']['inst']:<7} "
              f"{d['valid']['img']:>5}|{d['valid']['inst']:<7} "
              f"{d['test']['img']:>5}|{d['test']['inst']:<7}")
        for split in ("valid", "test"):
            n = d[split]["img"]
            if n == 0:
                continue
            if n < CRITICAL_EVAL:
                critical.append((cls, split, n))
            elif n < MIN_RELIABLE_EVAL:
                weak.append((cls, split, n))
    print()
    print(f"model-local id histogram: {dict(sorted(id_hist.items()))}")
    print(f"contains canonical id 8? {'YES (BAD)' if 8 in id_hist else 'no (correct)'}")
    print(f"contains counter class?  {'YES (BAD)' if 'counter' in per_class else 'no (correct)'}")
    print()

    print("Evaluation reliability")
    print("-" * 22)
    if critical:
        print(f"CRITICAL (< {CRITICAL_EVAL} eval images -> metric is noise):")
        for cls, split, n in critical:
            print(f"  {cls} {split}={n}")
    if weak:
        print(f"WEAK (< {MIN_RELIABLE_EVAL} eval images -> unstable metric):")
        for cls, split, n in weak:
            print(f"  {cls} {split}={n}")
    if not critical and not weak:
        print("None.")
    print()

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image", "class", "model_local_id", "canonical_id",
                                                    "line", "cx", "cy", "w", "h", "area", "aspect", "flags"])
        writer.writeheader()
        writer.writerows(suspicious)

    flag_counts = defaultdict(int)
    for s in suspicious:
        for f in s["flags"].split("|"):
            flag_counts[f] += 1
    print(f"Suspicious boxes (advisory, NOT deleted): {len(suspicious)} -> {REVIEW_PATH}")
    for flag, count in sorted(flag_counts.items()):
        print(f"  {flag}: {count}")
    print()

    print(f"cross-split exact-duplicate groups: {len(cross)}")
    print(f"unreadable images: {unreadable}")
    print(f"filename collisions: {sum(1 for e in errors if 'collision' in e)}")
    print(f"hard errors: {len(errors)}")
    for e in errors[:30]:
        print(f"  - {e}")
    print()
    if errors:
        print("VALIDATION: FAIL")
        raise SystemExit(1)
    print("VALIDATION: PASS (all hard checks clean; suspicious boxes advisory only)")


if __name__ == "__main__":
    main()
