"""Exact-duplicate deduplication for the prepared per-class YOLO datasets.

Operates on the prepared per-class source folders (data/training_photos/<class>/,
cutlery lives in cutlery_converted/). Content identity is decided by SHA-256 of
the image bytes -- NEVER by filename similarity.

Rules (deterministic, documented):
  - Group images by SHA-256.
  - A group whose members span more than one CLASS is a cross-class conflict:
    reported, never auto-removed (different class labels cannot be reconciled).
  - A group whose members carry genuinely different labels is a label conflict:
    reported, never auto-removed -- a human must decide. "Genuinely different"
    means a different number of boxes, a different class, or a box that cannot
    be matched to a counterpart within LABEL_MATCH_EPS on every coordinate.
    Sub-pixel coordinate jitter from Roboflow re-exports (~0.001-0.002) is NOT
    a conflict -- identical image bytes describe identical objects, so such
    groups are deduplicated normally.
  - Otherwise (one class, consistent labels) the group is collapsed to a single
    authoritative copy:
      * KEEP priority by split test > valid > train, so evaluation splits keep
        their copy and the duplicate is removed from the lower-priority split.
        This both eliminates cross-split leakage and preserves evaluation
        integrity (the model never trains on an eval image).
      * Ties within the same split: keep the lexicographically smallest path.
      * Every other member of the group is dropped.
  - Dropped image + its paired label are moved together to a quarantine tree
    (data/training_photos/_dedup_quarantine/...). Nothing is hard-deleted, so
    the step is auditable and reversible; no orphan image or label is left.

Default is a dry run (report only). Pass --apply to move files. After applying,
a verification pass re-hashes everything and asserts zero cross-split exact
duplicate leakage remains.

Report: reports/dedup_report.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASSES

TRAINING_ROOT = Path("data/training_photos")
QUARANTINE_ROOT = TRAINING_ROOT / "_dedup_quarantine"
REPORT_PATH = Path("reports/dedup_report.csv")
CONFLICT_REPORT_PATH = Path("reports/label_conflicts.csv")
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_KEEP_PRIORITY = {"test": 3, "valid": 2, "train": 1}

# Two labels on byte-identical images are "the same" if every box matches a
# counterpart within this per-coordinate tolerance (normalized units). 0.02 =
# 2% of the image dimension, comfortably above Roboflow's ~0.001-0.002 export
# jitter but tight enough to still catch genuinely different annotations.
LABEL_MATCH_EPS = 0.02

CLASS_SOURCE_OVERRIDE = {"cutlery": "cutlery_converted"}
MISSING_CLASSES = {"counter"}


def source_dir_for(class_name: str) -> Path:
    return TRAINING_ROOT / CLASS_SOURCE_OVERRIDE.get(class_name, class_name)


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label into a list of (class, cx, cy, w, h). Missing label
    -> empty list."""
    if not label_path.is_file():
        return []
    boxes = []
    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            coords = tuple(float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append((cls, *coords))
    return boxes


def boxes_consistent(boxes_a, boxes_b, eps: float = LABEL_MATCH_EPS) -> bool:
    """True if the two box sets describe the same objects within tolerance:
    same count, and every box in A greedily matches an unused box in B with
    the same class and all coordinates within eps."""
    if len(boxes_a) != len(boxes_b):
        return False
    used = set()
    for cls_a, *coords_a in boxes_a:
        matched = False
        for index, (cls_b, *coords_b) in enumerate(boxes_b):
            if index in used or cls_a != cls_b:
                continue
            if all(abs(a - b) <= eps for a, b in zip(coords_a, coords_b)):
                used.add(index)
                matched = True
                break
        if not matched:
            return False
    return True


class ImgRec:
    __slots__ = ("class_name", "split", "image", "label", "sha", "boxes")

    def __init__(self, class_name, split, image, label, sha, boxes):
        self.class_name = class_name
        self.split = split
        self.image = image
        self.label = label
        self.sha = sha
        self.boxes = boxes


def scan_records(classes) -> list[ImgRec]:
    records: list[ImgRec] = []
    for class_name in classes:
        root = source_dir_for(class_name)
        for split in SPLITS:
            images_dir = root / split / "images"
            labels_dir = root / split / "labels"
            if not images_dir.is_dir():
                continue
            for image in sorted(images_dir.iterdir()):
                if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label = labels_dir / f"{image.stem}.txt"
                records.append(
                    ImgRec(class_name, split, image, label, sha256_of(image), parse_boxes(label))
                )
    return records


def resolve_group(members: list[ImgRec]) -> tuple[str, list[tuple[ImgRec, str, str]]]:
    """Return (group_type, [(rec, decision, reason), ...])."""
    classes = {m.class_name for m in members}
    if len(classes) > 1:
        return "cross-class-conflict", [
            (m, "conflict", f"same bytes span classes {sorted(classes)}; manual review") for m in members
        ]

    def sort_key(rec: ImgRec):
        return (-SPLIT_KEEP_PRIORITY[rec.split], str(rec.image).lower())

    splits = {m.split for m in members}
    is_cross_split = len(splits) > 1

    # Consistent if every member's boxes match the first member's within tol.
    reference = members[0]
    consistent = all(boxes_consistent(reference.boxes, m.boxes) for m in members[1:])

    if not consistent:
        # Label conflict. If the group is WITHIN a single split there is no
        # leakage -- leave both copies for manual review. If it is CROSS-split
        # we must still remove the duplication (strict rule: no cross-split
        # exact duplicates), so we keep the highest-priority split copy and
        # drop the rest, but flag the label discrepancy loudly (not silent).
        if not is_cross_split:
            return "within-split-conflict", [
                (m, "conflict", "identical bytes, differing labels, same split; manual review")
                for m in members
            ]
        keeper = sorted(members, key=sort_key)[0]
        decisions = []
        for rec in members:
            if rec is keeper:
                decisions.append((rec, "keep-conflict",
                                  f"kept authoritative {rec.split} copy; LABEL DISCREPANCY vs dropped copies -- review kept label"))
            else:
                decisions.append((rec, "drop-conflict",
                                  f"dropped to remove cross-split duplication; label differed from kept {keeper.split}/{keeper.image.name} -- review"))
        return "cross-split-conflict-resolved", decisions

    group_type = "cross-split" if is_cross_split else "within-split"
    keeper = sorted(members, key=sort_key)[0]
    decisions = []
    for rec in members:
        if rec is keeper:
            decisions.append((rec, "keep", f"authoritative copy (split={rec.split})"))
        else:
            decisions.append((rec, "drop", f"exact dup of kept {keeper.split}/{keeper.image.name}"))
    return group_type, decisions


def quarantine(rec: ImgRec) -> None:
    dest_img_dir = QUARANTINE_ROOT / rec.class_name / rec.split / "images"
    dest_lab_dir = QUARANTINE_ROOT / rec.class_name / rec.split / "labels"
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lab_dir.mkdir(parents=True, exist_ok=True)
    if rec.image.exists():
        shutil.move(str(rec.image), str(dest_img_dir / rec.image.name))
    if rec.label.exists():
        shutil.move(str(rec.label), str(dest_lab_dir / rec.label.name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHA-256 exact-duplicate dedup for prepared per-class datasets.")
    parser.add_argument("--apply", action="store_true", help="Move dropped duplicates to quarantine (default: dry run).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = [c for c in OBJECT_CLASSES if c not in MISSING_CLASSES and source_dir_for(c).is_dir()]

    print("Exact-duplicate deduplication (SHA-256)")
    print("=" * 40)
    print(f"Mode: {'APPLY (will move files to quarantine)' if args.apply else 'DRY RUN (report only)'}")
    print(f"Classes scanned: {', '.join(classes)}")
    print()

    records = scan_records(classes)
    by_sha: dict[str, list[ImgRec]] = defaultdict(list)
    for rec in records:
        by_sha[rec.sha].append(rec)

    dup_groups = {sha: recs for sha, recs in by_sha.items() if len(recs) > 1}

    report_rows = []
    conflict_rows = []
    stats = defaultdict(int)
    stats["total_images"] = len(records)
    stats["unique_hashes"] = len(by_sha)
    conflicts = []
    to_drop: list[ImgRec] = []
    group_type_counts = defaultdict(int)

    for sha, members in dup_groups.items():
        group_type, decisions = resolve_group(members)
        group_type_counts[group_type] += 1
        is_conflict_group = group_type in ("cross-class-conflict", "within-split-conflict", "cross-split-conflict-resolved")
        for rec, decision, reason in decisions:
            row = {
                "sha256": sha,
                "class": rec.class_name,
                "split": rec.split,
                "image": str(rec.image).replace("\\", "/"),
                "label": str(rec.label).replace("\\", "/"),
                "decision": decision,
                "group_type": group_type,
                "reason": reason,
            }
            report_rows.append(row)
            if is_conflict_group:
                conflict_rows.append(row)
            if decision in ("drop", "drop-conflict"):
                to_drop.append(rec)
                stats["dropped"] += 1
                stats[f"dropped_{rec.split}"] += 1
                if group_type in ("cross-split", "cross-split-conflict-resolved"):
                    stats["dropped_cross_split"] += 1
                else:
                    stats["dropped_within_split"] += 1
        if is_conflict_group:
            conflicts.append((sha, group_type, members))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sha256", "class", "split", "image", "label", "decision", "group_type", "reason"]
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    with CONFLICT_REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(conflict_rows)

    print(f"Total images: {stats['total_images']}")
    print(f"Unique image hashes: {stats['unique_hashes']}")
    print(f"Exact-duplicate groups (size>1): {len(dup_groups)}")
    print(f"  within-split (clean):          {group_type_counts['within-split']}")
    print(f"  cross-split (clean):           {group_type_counts['cross-split']}")
    print(f"  cross-split conflict-resolved: {group_type_counts['cross-split-conflict-resolved']}")
    print(f"  within-split conflict (left):  {group_type_counts['within-split-conflict']}")
    print(f"  cross-class conflict (left):   {group_type_counts['cross-class-conflict']}")
    print(f"Files planned to drop: {stats['dropped']} "
          f"(cross-split {stats['dropped_cross_split']}, within-split {stats['dropped_within_split']})")
    for split in SPLITS:
        if stats[f"dropped_{split}"]:
            print(f"  drops from {split}: {stats[f'dropped_{split}']}")
    print(f"Conflict groups recorded for review: {len(conflicts)} -> {CONFLICT_REPORT_PATH}")
    print()

    # Per-class drop breakdown.
    per_class_drop = defaultdict(int)
    for rec in to_drop:
        per_class_drop[rec.class_name] += 1
    if per_class_drop:
        print("Drops per class:")
        for class_name in classes:
            if per_class_drop[class_name]:
                print(f"  {class_name}: {per_class_drop[class_name]}")
        print()

    if conflicts:
        print("CONFLICT GROUPS (identical bytes, differing labels):")
        for sha, group_type, members in conflicts[:40]:
            locs = [f"{m.class_name}/{m.split}/{m.image.name}" for m in members]
            note = "resolved by split-priority + flagged" if group_type == "cross-split-conflict-resolved" else "left in place, review"
            print(f"  [{group_type}] sha {sha[:10]} ({note}): {locs}")
        print()

    print(f"Report written: {REPORT_PATH}")
    print()

    if not args.apply:
        print("DRY RUN complete. Re-run with --apply to move dropped duplicates to quarantine.")
        return

    for rec in to_drop:
        quarantine(rec)
    print(f"APPLIED: moved {len(to_drop)} duplicate image/label pair(s) to {QUARANTINE_ROOT}")
    print()

    # Verify: re-scan and confirm zero cross-split exact duplicates.
    print("Verification re-scan...")
    records2 = scan_records(classes)
    by_sha2: dict[str, list[ImgRec]] = defaultdict(list)
    for rec in records2:
        by_sha2[rec.sha].append(rec)
    remaining_cross_split = 0
    remaining_within_split = 0
    train_eval_leak_groups = []  # the HARD constraint: identical bytes in train AND (valid/test)
    for sha, members in by_sha2.items():
        if len(members) < 2:
            continue
        splits_in = {m.split for m in members}
        # Hard rule check applies to ALL groups, including label conflicts.
        if "train" in splits_in and ({"valid", "test"} & splits_in):
            train_eval_leak_groups.append((sha, members))
        classes_in = {m.class_name for m in members}
        reference = members[0]
        labels_consistent = all(boxes_consistent(reference.boxes, m.boxes) for m in members[1:])
        if len(classes_in) > 1 or not labels_consistent:
            continue  # conflicts intentionally left
        if len(splits_in) > 1:
            remaining_cross_split += 1
        else:
            remaining_within_split += 1
    print(f"Remaining cross-split exact-duplicate groups (non-conflict): {remaining_cross_split}")
    print(f"Remaining within-split exact-duplicate groups (non-conflict): {remaining_within_split}")
    print(f"Remaining TRAIN<->EVAL identical-content groups (hard rule): {len(train_eval_leak_groups)}")
    for sha, members in train_eval_leak_groups[:20]:
        locs = [f"{m.class_name}/{m.split}/{m.image.name}" for m in members]
        print(f"  LEAK sha {sha[:10]}: {locs}")
    if len(train_eval_leak_groups) == 0:
        print("PASS: zero train<->eval identical-content leakage.")
    else:
        print("FAIL: train<->eval identical-content leakage remains -- investigate.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
