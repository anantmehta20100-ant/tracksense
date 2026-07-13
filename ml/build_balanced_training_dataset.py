"""Build the class-balanced 8-class TRAINING dataset.

Reads  data/training_8class           (model-local ids 0..7, counter excluded)
Writes data/training_8class_balanced  (final training candidate)

Strategy (combination; justified against actual post-dedup counts):
  - CAP the two dominant classes' TRAIN images by deterministic random sample.
  - KEEP every train image of every other class.
  - AUGMENT the minority classes' TRAIN sets with conservative, bbox-correct
    transforms (ml/augment.py) until they reach their target count.

Hard guarantees:
  - VALID and TEST are copied byte-for-byte from the 8-class source. Their
    distribution is never rebalanced and never augmented.
  - Augmented images are derived only from TRAIN originals and written only to
    TRAIN, so no augmented pixel can ever reach valid/test.
  - Deterministic: one seed drives both the capping sample and every transform.
  - Provenance recorded per augmented image: origin filename + transform string.

Outputs reports/balanced_build_manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.augment import augment
from ml.class_schema import MODEL_LOCAL_NAMES, NUM_TRAINING_CLASSES

SRC_ROOT = Path("data/training_8class")
DST_ROOT = Path("data/training_8class_balanced")
MANIFEST_PATH = Path("reports/balanced_build_manifest.csv")
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SEED = 42

# Dominant classes: cap TRAIN images (random sample, seeded).
TRAIN_CAPS = {"plate": 2800, "cutlery": 2800}

# Minority classes: keep all originals, augment up to this TRAIN image count.
TRAIN_AUG_TARGETS = {"bowl": 480, "chopping_board": 480, "nut_butter_jar": 310}


def class_of(image_name: str) -> str:
    return image_name.split("__", 1)[0]


def read_boxes(label_path: Path):
    boxes = []
    if not label_path.is_file():
        return boxes
    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        boxes.append((int(float(parts[0])), *(float(v) for v in parts[1:5])))
    return boxes


def write_boxes(label_path: Path, boxes) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (c, cx, cy, w, h) in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def reset_dst() -> None:
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    for split in SPLITS:
        (DST_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (DST_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_verbatim(split: str, manifest: list) -> dict:
    counts = defaultdict(int)
    src_images = SRC_ROOT / split / "images"
    src_labels = SRC_ROOT / split / "labels"
    for image in sorted(src_images.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        shutil.copy2(image, DST_ROOT / split / "images" / image.name)
        label = src_labels / f"{image.stem}.txt"
        if label.is_file():
            shutil.copy2(label, DST_ROOT / split / "labels" / label.name)
        cls = class_of(image.name)
        counts[cls] += 1
        manifest.append({
            "split": split, "image": image.name, "class": cls,
            "source_type": "verbatim", "origin_image": "", "transform": "", "included": True,
        })
    return counts


def build_train(rng: random.Random, np_rng: np.random.Generator, manifest: list):
    src_images = SRC_ROOT / "train" / "images"
    src_labels = SRC_ROOT / "train" / "labels"
    dst_images = DST_ROOT / "train" / "images"
    dst_labels = DST_ROOT / "train" / "labels"

    by_class: dict[str, list[Path]] = defaultdict(list)
    for image in sorted(src_images.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            by_class[class_of(image.name)].append(image)

    before = {c: len(v) for c, v in by_class.items()}
    kept = defaultdict(int)
    augmented = defaultdict(int)

    for cls in sorted(by_class):
        originals = sorted(by_class[cls], key=lambda p: p.name)
        if cls in TRAIN_CAPS and len(originals) > TRAIN_CAPS[cls]:
            selected = sorted(rng.sample(originals, TRAIN_CAPS[cls]), key=lambda p: p.name)
        else:
            selected = originals

        for image in selected:
            shutil.copy2(image, dst_images / image.name)
            label = src_labels / f"{image.stem}.txt"
            if label.is_file():
                shutil.copy2(label, dst_labels / label.name)
            kept[cls] += 1
            manifest.append({
                "split": "train", "image": image.name, "class": cls,
                "source_type": "capped-kept" if cls in TRAIN_CAPS else "original",
                "origin_image": "", "transform": "", "included": True,
            })

        if cls in TRAIN_AUG_TARGETS:
            need = max(0, TRAIN_AUG_TARGETS[cls] - len(selected))
            for i in range(need):
                origin = selected[i % len(selected)]
                boxes = read_boxes(src_labels / f"{origin.stem}.txt")
                aug_image, aug_boxes, transform = augment(origin, boxes, rng, np_rng)
                stem = f"{cls}__aug{i:04d}__{origin.stem}"
                aug_image.save(dst_images / f"{stem}.jpg", quality=95)
                write_boxes(dst_labels / f"{stem}.txt", aug_boxes)
                augmented[cls] += 1
                manifest.append({
                    "split": "train", "image": f"{stem}.jpg", "class": cls,
                    "source_type": "augmented", "origin_image": origin.name,
                    "transform": transform, "included": True,
                })

    return before, dict(kept), dict(augmented)


def instances_per_class(split: str) -> dict:
    counts = defaultdict(int)
    for label in (DST_ROOT / split / "labels").glob("*.txt"):
        cls = class_of(label.name)
        counts[cls] += sum(1 for l in label.read_text().splitlines() if l.strip())
    return counts


def ratio(counts: dict) -> float:
    present = [v for v in counts.values() if v > 0]
    return max(present) / min(present) if present else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the balanced 8-class TRAIN dataset.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not (SRC_ROOT / "train" / "images").is_dir():
        raise SystemExit(f"{SRC_ROOT} not found. Run: python -m ml.build_8class_dataset")

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    print("Building balanced 8-class training dataset")
    print("=" * 42)
    print(f"seed={args.seed}")
    print(f"caps           : {TRAIN_CAPS}")
    print(f"augment targets: {TRAIN_AUG_TARGETS}\n")

    reset_dst()
    manifest: list = []
    before, kept, aug = build_train(rng, np_rng, manifest)
    valid_counts = copy_verbatim("valid", manifest)
    test_counts = copy_verbatim("test", manifest)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image", "class", "source_type", "origin_image", "transform", "included"])
        writer.writeheader()
        writer.writerows(manifest)

    after = {c: kept.get(c, 0) + aug.get(c, 0) for c in set(kept) | set(aug)}

    print("Actions (TRAIN only)")
    print("-" * 20)
    for local in range(NUM_TRAINING_CLASSES):
        cls = MODEL_LOCAL_NAMES[local]
        if cls not in before:
            continue
        if cls in TRAIN_CAPS:
            pct = 100.0 * kept.get(cls, 0) / before[cls]
            print(f"  {cls:16} CAP     {before[cls]:>5} -> {kept.get(cls,0):>5}  ({pct:.1f}% retained)")
        elif cls in TRAIN_AUG_TARGETS:
            print(f"  {cls:16} AUGMENT {before[cls]:>5} + {aug.get(cls,0):>4} augmented = {after.get(cls,0):>5}")
        else:
            print(f"  {cls:16} KEEP ALL {before[cls]:>4}")
    print()

    print("Train images per class (before -> after)")
    print("-" * 40)
    for local in range(NUM_TRAINING_CLASSES):
        cls = MODEL_LOCAL_NAMES[local]
        print(f"  {local} {cls:16} {before.get(cls,0):>6} -> {after.get(cls,0):>6}")
    print()

    tr_inst = instances_per_class("train")
    print(f"Train IMAGE   imbalance  before {ratio(before):.1f}x  ->  after {ratio(after):.1f}x")
    print(f"Train INSTANCE imbalance after: {ratio(tr_inst):.1f}x")
    print()
    print(f"train images: {sum(after.values())}  (augmented: {sum(aug.values())})")
    print(f"valid images: {sum(valid_counts.values())} (verbatim)")
    print(f"test  images: {sum(test_counts.values())} (verbatim)")
    print(f"manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
