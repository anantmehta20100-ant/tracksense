"""Advisory pHash near-duplicate audit for a YOLO dataset.

Images are compared across train/valid/test within each class. Perceptual
similarity is only a review signal: this script never moves or deletes data.

Defaults preserve the historical unified-dataset audit. Pass --root and
--report to inspect a final training candidate and write a separate report.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASS_TO_ID

DEFAULT_ROOT = Path("data/training_photos/unified")
DEFAULT_REPORT_PATH = Path("reports/near_duplicate_review.csv")
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PHASH_DISTANCE_THRESHOLD = 6
PHASH_SIZE = 32
PHASH_LOWFREQ = 8

_k = np.arange(PHASH_SIZE)
_DCT = np.sqrt(2.0 / PHASH_SIZE) * np.cos(
    np.pi * (2 * _k[None, :] + 1) * _k[:, None] / (2 * PHASH_SIZE)
)
_DCT[0, :] /= np.sqrt(2.0)


def phash(path: Path) -> int | None:
    try:
        with Image.open(path) as image:
            gray = image.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.BILINEAR)
            array = np.asarray(gray, dtype=np.float64)
    except Exception:
        return None

    dct = _DCT @ array @ _DCT.T
    block = dct[:PHASH_LOWFREQ, :PHASH_LOWFREQ].flatten()
    median = np.median(block[1:])
    bits = 0
    for index, value in enumerate(block):
        if value > median:
            bits |= 1 << index
    return bits


def class_of(image_name: str) -> str:
    return image_name.split("__", 1)[0]


def collect(root: Path) -> dict:
    """Return class -> split -> [(image name, pHash)] for a dataset root."""
    data: dict = defaultdict(lambda: defaultdict(list))
    unreadable = 0
    for split in SPLITS:
        images_dir = root / split / "images"
        if not images_dir.is_dir():
            continue
        for image in sorted(images_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_hash = phash(image)
            if image_hash is None:
                unreadable += 1
                continue
            data[class_of(image.name)][split].append((image.name, image_hash))
    if unreadable:
        print(f"WARNING: {unreadable} image(s) could not be hashed.")
    return data


def review_priority(distance: int) -> str:
    if distance == 0:
        return "strongest"
    if distance <= 2:
        return "strong"
    return "advisory"


def audit_dataset(root: Path, report_path: Path) -> dict[str, int]:
    """Write advisory cross-split pHash matches and return summary counts."""
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    print("Perceptual near-duplicate audit (pHash, advisory)")
    print("=" * 50)
    print(f"Dataset root: {root}")
    print(f"Report path: {report_path}")
    print(f"Distance threshold (Hamming <=): {PHASH_DISTANCE_THRESHOLD}")
    print()

    data = collect(root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    per_class_counts = {}
    cross_pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))

    for class_name in sorted(data, key=lambda name: OBJECT_CLASS_TO_ID.get(name, 99)):
        canonical_id = OBJECT_CLASS_TO_ID.get(class_name, -1)
        count = 0
        for split_a, split_b in cross_pairs:
            group_a = data[class_name].get(split_a, [])
            group_b = data[class_name].get(split_b, [])
            for name_a, hash_a in group_a:
                for name_b, hash_b in group_b:
                    distance = (hash_a ^ hash_b).bit_count()
                    if distance <= PHASH_DISTANCE_THRESHOLD:
                        rows.append({
                            "path_a": (Path(split_a) / "images" / name_a).as_posix(),
                            "path_b": (Path(split_b) / "images" / name_b).as_posix(),
                            "split_a": split_a,
                            "split_b": split_b,
                            "class_name": class_name,
                            "canonical_id": canonical_id,
                            "hash_distance": distance,
                            "is_cross_split": "yes",
                            "review_priority": review_priority(distance),
                        })
                        count += 1
        per_class_counts[class_name] = count

    rows.sort(key=lambda row: (
        row["hash_distance"],
        row["canonical_id"],
        row["path_a"],
        row["path_b"],
    ))
    fieldnames = [
        "path_a",
        "path_b",
        "split_a",
        "split_b",
        "class_name",
        "canonical_id",
        "hash_distance",
        "is_cross_split",
        "review_priority",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cross_split_pairs = sum(row["is_cross_split"] == "yes" for row in rows)
    strongest_matches = sum(row["review_priority"] == "strongest" for row in rows)

    print("Suspicious cross-split near-duplicate pairs per class (advisory):")
    print(f"{'class':16} {'pairs':>8}")
    print("-" * 28)
    for class_name in sorted(per_class_counts, key=lambda name: OBJECT_CLASS_TO_ID.get(name, 99)):
        print(f"{class_name:16} {per_class_counts[class_name]:>8}")
    print()
    print(f"Total suspicious pairs: {len(rows)}")
    print(f"Cross-split suspicious pairs: {cross_split_pairs}")
    print(f"Strongest matches (distance 0): {strongest_matches}")
    print(f"Report written to: {report_path}")
    if strongest_matches:
        print("Strongest matches for manual review:")
        for row in (row for row in rows if row["review_priority"] == "strongest"):
            print(f"  {row['path_a']} <-> {row['path_b']} ({row['class_name']})")
    print()
    print("NOTE: advisory only. No images were removed or moved.")

    return {
        "total_suspicious_pairs": len(rows),
        "cross_split_suspicious_pairs": cross_split_pairs,
        "strongest_matches": strongest_matches,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advisory pHash near-duplicate audit for a YOLO dataset.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="YOLO dataset root containing train/valid/test images.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="CSV path for advisory review rows.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    audit_dataset(Path(args.root), Path(args.report))


if __name__ == "__main__":
    main()
