"""Strong validation for prepared per-class YOLO datasets.

For each requested class this checks, per split (train/valid/test):
  - every image has a matching label and every label a matching image
  - every annotation's class id equals the expected global TrackSense id
  - YOLO coordinates are numeric and within normalized [0, 1] ranges
  - empty and unparseable/corrupt label files
  - unreadable/corrupt images (verified via PIL)
  - images with zero accepted annotations (empty label files)
  - exact-content duplicate images (SHA-256), within and across splits
  - duplicate image basenames
  - perceptual near-duplicate images across splits (8x8 average hash,
    Hamming distance <= PERCEPTUAL_HAMMING_THRESHOLD) -- real image content
    comparison, not just filename matching

Exit code is non-zero if any hard error (pairing / class-id / coordinate /
corrupt) is found. Duplicates and cross-split leakage are reported as
warnings, not hard failures, so the caller can decide.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASS_TO_ID, OBJECT_CLASSES

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TRAINING_ROOT = Path("data/training_photos")
PERCEPTUAL_HAMMING_THRESHOLD = 4  # <= this many differing bits = near-duplicate


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def average_hash(path: Path) -> int | None:
    """64-bit perceptual average hash. Returns None if the image can't be read."""
    try:
        with Image.open(path) as img:
            small = img.convert("L").resize((8, 8), Image.BILINEAR)
            pixels = list(small.getdata())
    except Exception:
        return None
    mean = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value >= mean:
            bits |= 1 << index
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def list_split_files(class_root: Path, split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    images_dir = class_root / split / "images"
    labels_dir = class_root / split / "labels"
    images = {
        path.stem: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    } if images_dir.is_dir() else {}
    labels = {
        path.stem: path
        for path in labels_dir.iterdir()
        if path.is_file() and path.name != ".gitkeep" and path.suffix.lower() == ".txt"
    } if labels_dir.is_dir() else {}
    return images, labels


def validate_class(class_name: str) -> dict:
    expected_id = OBJECT_CLASS_TO_ID[class_name]
    class_root = TRAINING_ROOT / class_name

    errors: list[str] = []
    warnings: list[str] = []
    per_split_counts: dict[str, dict[str, int]] = {}

    content_hashes: dict[str, list[str]] = defaultdict(list)          # sha -> ["split/name"]
    basename_index: dict[str, list[str]] = defaultdict(list)          # basename -> ["split/name"]
    perceptual_by_split: dict[str, list[tuple[str, int]]] = defaultdict(list)  # split -> [(desc, ahash)]
    total_instances = 0

    print(f"### Class '{class_name}' (expected global id {expected_id})")
    for split in SPLITS:
        images, labels = list_split_files(class_root, split)

        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        if missing_labels:
            errors.append(f"{class_name}/{split}: {len(missing_labels)} image(s) without a label: {missing_labels[:5]}")
        if missing_images:
            errors.append(f"{class_name}/{split}: {len(missing_images)} label(s) without an image: {missing_images[:5]}")

        split_instances = 0
        empty_labels = 0
        for stem, label_path in labels.items():
            lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
            non_empty = [ln for ln in lines if ln.strip()]
            if not non_empty:
                empty_labels += 1
                continue
            for line_number, raw in enumerate(non_empty, start=1):
                parts = raw.split()
                if len(parts) < 5:
                    errors.append(f"{label_path}:{line_number}: fewer than 5 fields: {raw!r}")
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    errors.append(f"{label_path}:{line_number}: non-numeric class id {parts[0]!r}")
                    continue
                if class_id != expected_id:
                    errors.append(f"{label_path}:{line_number}: class id {class_id}, expected {expected_id}")
                try:
                    coords = [float(v) for v in parts[1:5]]
                except ValueError:
                    errors.append(f"{label_path}:{line_number}: non-numeric coordinates {parts[1:5]}")
                    continue
                if any(not (0.0 <= v <= 1.0) for v in coords):
                    errors.append(f"{label_path}:{line_number}: coordinate outside [0,1]: {coords}")
                split_instances += 1

        if empty_labels:
            warnings.append(f"{class_name}/{split}: {empty_labels} empty label file(s) (zero-annotation images)")

        # Image readability + hashing.
        unreadable = 0
        for stem, image_path in images.items():
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as exc:
                errors.append(f"{image_path}: unreadable/corrupt image ({exc})")
                unreadable += 1
                continue
            descriptor = f"{split}/{image_path.name}"
            content_hashes[sha256_of(image_path)].append(descriptor)
            basename_index[image_path.name.lower()].append(descriptor)
            ahash = average_hash(image_path)
            if ahash is not None:
                perceptual_by_split[split].append((descriptor, ahash))

        per_split_counts[split] = {
            "images": len(images),
            "labels": len(labels),
            "instances": split_instances,
            "empty_labels": empty_labels,
            "unreadable": unreadable,
        }
        total_instances += split_instances
        print(f"  {split}: images={len(images)} labels={len(labels)} instances={split_instances} "
              f"empty_labels={empty_labels} unreadable={unreadable}")

    # Exact-content duplicates.
    exact_dupes = {sha: locs for sha, locs in content_hashes.items() if len(locs) > 1}
    for sha, locs in exact_dupes.items():
        cross_split = len({loc.split("/", 1)[0] for loc in locs}) > 1
        tag = "CROSS-SPLIT " if cross_split else ""
        warnings.append(f"{class_name}: {tag}exact-duplicate image (sha {sha[:10]}): {locs}")

    # Duplicate basenames.
    dupe_names = {name: locs for name, locs in basename_index.items() if len(locs) > 1}
    for name, locs in dupe_names.items():
        warnings.append(f"{class_name}: duplicate basename {name!r}: {locs}")

    # Perceptual near-duplicates across DIFFERENT splits (train/valid/test
    # leakage). Grouped by split so only cross-split pairs are compared.
    near_dupe_pairs = 0
    split_pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
    for split_a, split_b in split_pairs:
        group_a = perceptual_by_split.get(split_a, [])
        group_b = perceptual_by_split.get(split_b, [])
        for desc_a, hash_a in group_a:
            for desc_b, hash_b in group_b:
                if hamming(hash_a, hash_b) <= PERCEPTUAL_HAMMING_THRESHOLD:
                    near_dupe_pairs += 1
                    if near_dupe_pairs <= 15:
                        warnings.append(
                            f"{class_name}: perceptual near-duplicate across splits "
                            f"({desc_a} ~ {desc_b}, hamming<= {PERCEPTUAL_HAMMING_THRESHOLD})"
                        )
    if near_dupe_pairs > 15:
        warnings.append(f"{class_name}: ... {near_dupe_pairs - 15} more cross-split near-duplicate pairs")

    return {
        "class_name": class_name,
        "expected_id": expected_id,
        "per_split": per_split_counts,
        "total_instances": total_instances,
        "errors": errors,
        "warnings": warnings,
        "exact_dupe_groups": len(exact_dupes),
        "cross_split_near_dupes": near_dupe_pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strong validation for prepared per-class YOLO datasets.")
    parser.add_argument(
        "classes",
        nargs="*",
        help="Class names to validate. Default: all classes with a prepared folder.",
    )
    return parser.parse_args()


def prepared_classes() -> list[str]:
    result = []
    for class_name in OBJECT_CLASSES:
        class_root = TRAINING_ROOT / class_name
        has_files = any(
            (class_root / split / "images").is_dir()
            and any(
                p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                for p in (class_root / split / "images").iterdir()
            )
            for split in SPLITS
        )
        if has_files:
            result.append(class_name)
    return result


def main() -> None:
    args = parse_args()
    classes = args.classes or prepared_classes()

    print("Strong dataset validation")
    print("=" * 25)
    print(f"Classes: {', '.join(classes)}")
    print()

    reports = [validate_class(class_name) for class_name in classes]
    print()

    print("Summary")
    print("=" * 7)
    any_errors = False
    for report in reports:
        counts = report["per_split"]
        totals = {
            k: sum(counts.get(s, {}).get(k, 0) for s in SPLITS)
            for k in ("images", "labels", "instances")
        }
        status = "PASS" if not report["errors"] else f"FAIL ({len(report['errors'])} errors)"
        print(
            f"- {report['class_name']} (id {report['expected_id']}): "
            f"images={totals['images']} labels={totals['labels']} instances={totals['instances']} "
            f"exact_dupe_groups={report['exact_dupe_groups']} "
            f"cross_split_near_dupes={report['cross_split_near_dupes']} -> {status}"
        )
        if report["errors"]:
            any_errors = True

    print()
    for report in reports:
        if report["warnings"]:
            print(f"Warnings for {report['class_name']}:")
            for w in report["warnings"][:40]:
                print(f"  - {w}")
            if len(report["warnings"]) > 40:
                print(f"  - ... {len(report['warnings']) - 40} more warnings")
            print()

    for report in reports:
        if report["errors"]:
            print(f"ERRORS for {report['class_name']}:")
            for e in report["errors"][:40]:
                print(f"  - {e}")
            if len(report["errors"]) > 40:
                print(f"  - ... {len(report['errors']) - 40} more errors")
            print()

    if any_errors:
        print("VALIDATION FAILED: hard errors present (see above).")
        raise SystemExit(1)
    print("VALIDATION PASSED: no hard errors. Review warnings above for duplicates/leakage.")


if __name__ == "__main__":
    main()
