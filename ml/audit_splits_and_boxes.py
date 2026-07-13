"""Split-quality audit + strict label/box sanity for a YOLO dataset root.

Runs on the final training candidate (default: the balanced dataset).

Split quality (Phase 9): per class, train/valid/test image and instance counts;
flags classes whose valid or test set is too small for reliable metrics.

Label/box sanity (Phase 10): every label line is checked for
  - valid class id (0..8)
  - numeric coords, no NaN / inf
  - x_center, y_center in [0,1]
  - width, height in (0,1]
  - non-zero area
Plus advisory "suspicious" boxes (reported, never auto-deleted):
  - near-full-frame (area >= FULL_FRAME_AREA)
  - extremely tiny (area <= TINY_AREA)
  - extreme aspect ratio (>= EXTREME_AR)
Also: unreadable images, orphan images, orphan labels.

Hard problems -> non-zero exit. Suspicious boxes -> reports/box_sanity_review.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASSES, OBJECT_ID_TO_CLASS

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REVIEW_PATH = Path("reports/box_sanity_review.csv")

VALID_CLASS_IDS = set(OBJECT_ID_TO_CLASS)

# Eval-set size thresholds for the reliability flags.
MIN_RELIABLE_EVAL = 30      # < this -> weak
CRITICAL_EVAL = 10          # < this -> very unstable

# Suspicious-box thresholds (advisory).
FULL_FRAME_AREA = 0.98
TINY_AREA = 0.0005
EXTREME_AR = 20.0


def class_of(image_name: str) -> str:
    return image_name.split("__", 1)[0]


def audit(root: Path):
    hard_errors: list[str] = []
    suspicious: list[dict] = []
    per_class = defaultdict(lambda: {s: {"img": 0, "inst": 0} for s in SPLITS})
    unreadable = 0

    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS} if images_dir.is_dir() else {}
        labels = {p.stem: p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"} if labels_dir.is_dir() else {}

        for stem in sorted(set(images) - set(labels)):
            hard_errors.append(f"{split}: orphan image (no label): {images[stem].name}")
        for stem in sorted(set(labels) - set(images)):
            hard_errors.append(f"{split}: orphan label (no image): {labels[stem].name}")

        for stem, image_path in images.items():
            try:
                with Image.open(image_path) as im:
                    im.verify()
            except Exception as exc:
                hard_errors.append(f"{split}: unreadable image {image_path.name}: {exc}")
                unreadable += 1
            cls = class_of(image_path.name)
            per_class[cls][split]["img"] += 1

            label_path = labels.get(stem)
            if not label_path:
                continue
            for line_no, raw in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if not raw.strip():
                    continue
                parts = raw.split()
                if len(parts) < 5:
                    hard_errors.append(f"{label_path.name}:{line_no}: <5 fields")
                    continue
                try:
                    class_id = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    hard_errors.append(f"{label_path.name}:{line_no}: non-numeric")
                    continue
                if any(math.isnan(v) or math.isinf(v) for v in (cx, cy, w, h)):
                    hard_errors.append(f"{label_path.name}:{line_no}: NaN/inf coord")
                    continue
                if class_id not in VALID_CLASS_IDS:
                    hard_errors.append(f"{label_path.name}:{line_no}: invalid class id {class_id}")
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    hard_errors.append(f"{label_path.name}:{line_no}: center out of [0,1] ({cx},{cy})")
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    hard_errors.append(f"{label_path.name}:{line_no}: w/h not in (0,1] ({w},{h})")
                    continue
                per_class[cls][split]["inst"] += 1

                area = w * h
                aspect = max(w / h, h / w) if h > 0 and w > 0 else float("inf")
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
                        "line": line_no, "cx": cx, "cy": cy, "w": w, "h": h,
                        "area": round(area, 6), "aspect": round(aspect, 2),
                        "flags": "|".join(flags),
                    })

    return per_class, hard_errors, suspicious, unreadable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split quality + label/box sanity audit.")
    parser.add_argument("--root", default="data/training_photos/balanced", help="Dataset root to audit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    print(f"Auditing dataset root: {root}")
    print("=" * 60)

    per_class, hard_errors, suspicious, unreadable = audit(root)

    print("Per-class split counts (images | instances)")
    print(f"{'id':>2} {'class':16} {'train':>12} {'valid':>12} {'test':>12}")
    print("-" * 60)
    weak, critical = [], []
    for cid in range(len(OBJECT_CLASSES)):
        cls = OBJECT_ID_TO_CLASS[cid]
        d = per_class.get(cls, {s: {"img": 0, "inst": 0} for s in SPLITS})
        print(f"{cid:>2} {cls:16} "
              f"{d['train']['img']:>4}/{d['train']['inst']:<7} "
              f"{d['valid']['img']:>4}/{d['valid']['inst']:<7} "
              f"{d['test']['img']:>4}/{d['test']['inst']:<7}")
        if cls in ("counter",):
            continue
        for split in ("valid", "test"):
            n = d[split]["img"]
            if n == 0:
                continue
            if n < CRITICAL_EVAL:
                critical.append((cls, split, n))
            elif n < MIN_RELIABLE_EVAL:
                weak.append((cls, split, n))
    print()

    print("Evaluation-reliability flags")
    print("-" * 28)
    if critical:
        print(f"CRITICAL (< {CRITICAL_EVAL} eval images -> per-class metric essentially noise):")
        for cls, split, n in critical:
            print(f"  {cls} {split}={n}")
    if weak:
        print(f"WEAK (< {MIN_RELIABLE_EVAL} eval images -> unstable per-class metric):")
        for cls, split, n in weak:
            print(f"  {cls} {split}={n}")
    if not critical and not weak:
        print("None.")
    print()

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image", "class", "line", "cx", "cy", "w", "h", "area", "aspect", "flags"])
        writer.writeheader()
        writer.writerows(suspicious)

    susp_by_flag = defaultdict(int)
    for s in suspicious:
        for f in s["flags"].split("|"):
            susp_by_flag[f] += 1
    print("Suspicious boxes (advisory, NOT deleted)")
    print("-" * 40)
    print(f"Total: {len(suspicious)} -> {REVIEW_PATH}")
    for flag, count in sorted(susp_by_flag.items()):
        print(f"  {flag}: {count}")
    print()

    print(f"Unreadable images: {unreadable}")
    print(f"Hard errors: {len(hard_errors)}")
    for e in hard_errors[:30]:
        print(f"  - {e}")
    if len(hard_errors) > 30:
        print(f"  - ... {len(hard_errors) - 30} more")
    print()
    if hard_errors:
        print("LABEL/BOX SANITY: FAIL (hard errors present).")
        raise SystemExit(1)
    print("LABEL/BOX SANITY: PASS (no hard errors; suspicious boxes are advisory).")


if __name__ == "__main__":
    main()
