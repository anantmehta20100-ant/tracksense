"""Build the unified multi-class YOLO dataset from the prepared per-class
folders, without corrupting class IDs.

Each per-class folder under data/training_photos/<class>/ (cutlery lives in
cutlery_converted/) holds train/valid/test images+labels. This script:
  - copies every image into data/training_photos/unified/<split>/images,
    prefixing the filename with the class name (`<class>__<orig>`) so files
    from different classes can never collide;
  - rewrites every label's class id to that class's GLOBAL TrackSense id from
    config/allergens.py (cutlery_converted still carries local id 0, so this
    remap is mandatory, not cosmetic);
  - preserves each source's current split assignment (no resplitting);
  - verifies image<->label pairing and class ids after copying;
  - skips the missing counter class but keeps id 7 defined in data.yaml;
  - prints a class distribution table + imbalance analysis.

Reproducible: clears unified/ first, iterates classes/splits in a fixed
order, pure file copy (no randomness).
"""

from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASSES, OBJECT_CLASS_TO_ID

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TRAINING_ROOT = Path("data/training_photos")
UNIFIED_ROOT = TRAINING_ROOT / "unified"

# Classes whose prepared data lives in a differently-named folder.
CLASS_SOURCE_OVERRIDE = {
    "cutlery": "cutlery_converted",
}

# Classes with no data yet -- kept in data.yaml, excluded from the build.
MISSING_CLASSES = {"counter"}


def source_dir_for(class_name: str) -> Path:
    folder = CLASS_SOURCE_OVERRIDE.get(class_name, class_name)
    return TRAINING_ROOT / folder


def has_any_images(class_name: str) -> bool:
    root = source_dir_for(class_name)
    for split in SPLITS:
        images_dir = root / split / "images"
        if images_dir.is_dir() and any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in images_dir.iterdir()
        ):
            return True
    return False


def reset_unified() -> None:
    if UNIFIED_ROOT.exists():
        shutil.rmtree(UNIFIED_ROOT)
    for split in SPLITS:
        (UNIFIED_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (UNIFIED_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def remap_label_to_global(source_label: Path, destination_label: Path, global_id: int) -> int:
    """Rewrite every YOLO line's class id to global_id; keep geometry. Returns
    the number of boxes written."""
    lines_out: list[str] = []
    for raw in source_label.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            print(f"WARNING: skipping malformed label line in {source_label}: {raw!r}")
            continue
        parts[0] = str(global_id)
        lines_out.append(" ".join(parts))
    if lines_out:
        destination_label.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return len(lines_out)


def unique_destination(destination_dir: Path, filename: str) -> Path:
    candidate = destination_dir / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for index in range(1, 100000):
        renamed = destination_dir / f"{stem}__dup{index}{suffix}"
        if not renamed.exists():
            return renamed
    raise RuntimeError(f"Could not find a non-colliding name for {filename}")


def build_class(class_name: str, distribution: dict) -> None:
    global_id = OBJECT_CLASS_TO_ID[class_name]
    source_root = source_dir_for(class_name)

    for split in SPLITS:
        images_dir = source_root / split / "images"
        labels_dir = source_root / split / "labels"
        if not images_dir.is_dir():
            continue

        dest_images = UNIFIED_ROOT / split / "images"
        dest_labels = UNIFIED_ROOT / split / "labels"

        for source_image in sorted(images_dir.iterdir()):
            if not source_image.is_file() or source_image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            source_label = labels_dir / f"{source_image.stem}.txt"
            if not source_label.is_file():
                print(f"WARNING: no label for {source_image}; skipping (would be an unlabeled image).")
                continue

            prefixed_name = f"{class_name}__{source_image.name}"
            destination_image = unique_destination(dest_images, prefixed_name)
            destination_label = dest_labels / f"{destination_image.stem}.txt"

            boxes = remap_label_to_global(source_label, destination_label, global_id)
            if boxes == 0:
                # Empty label -> don't copy the image either (keeps pairs clean).
                continue

            shutil.copy2(source_image, destination_image)
            distribution[class_name][split]["images"] += 1
            distribution[class_name][split]["instances"] += boxes


def verify_unified() -> list[str]:
    errors: list[str] = []
    valid_ids = set(OBJECT_CLASS_TO_ID.values())

    for split in SPLITS:
        images_dir = UNIFIED_ROOT / split / "images"
        labels_dir = UNIFIED_ROOT / split / "labels"
        images = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {p.stem for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}

        for stem in sorted(images - labels):
            errors.append(f"{split}: image {stem} has no label")
        for stem in sorted(labels - images):
            errors.append(f"{split}: label {stem} has no image")

        for label_path in labels_dir.glob("*.txt"):
            # class id implied by the filename prefix must match the label content
            prefix = label_path.name.split("__", 1)[0]
            expected_id = OBJECT_CLASS_TO_ID.get(prefix)
            for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not raw.strip():
                    continue
                parts = raw.split()
                class_id = int(float(parts[0]))
                if class_id not in valid_ids:
                    errors.append(f"{label_path}:{line_number}: class id {class_id} not a valid TrackSense id")
                if expected_id is not None and class_id != expected_id:
                    errors.append(
                        f"{label_path}:{line_number}: class id {class_id} != {expected_id} implied by filename prefix {prefix!r}"
                    )
                coords = [float(v) for v in parts[1:5]]
                if any(not (0.0 <= v <= 1.0) for v in coords):
                    errors.append(f"{label_path}:{line_number}: coordinate outside [0,1]: {coords}")
    return errors


def print_distribution(distribution: dict) -> None:
    print("Class distribution (unified dataset)")
    print("=" * 36)
    header = f"{'id':>2}  {'class':<16} {'train_img':>9} {'valid_img':>9} {'test_img':>8} {'instances':>10}"
    print(header)
    print("-" * len(header))

    totals_by_class: dict[str, int] = {}
    for class_name in OBJECT_CLASSES:
        global_id = OBJECT_CLASS_TO_ID[class_name]
        per = distribution.get(class_name, {})
        train_img = per.get("train", {}).get("images", 0)
        valid_img = per.get("valid", {}).get("images", 0)
        test_img = per.get("test", {}).get("images", 0)
        instances = sum(per.get(s, {}).get("instances", 0) for s in SPLITS)
        totals_by_class[class_name] = instances
        flag = "  <-- MISSING (0 samples)" if class_name in MISSING_CLASSES else ""
        print(f"{global_id:>2}  {class_name:<16} {train_img:>9} {valid_img:>9} {test_img:>8} {instances:>10}{flag}")

    print()
    return totals_by_class


def print_imbalance(distribution: dict, totals_by_class: dict[str, int]) -> None:
    present = {c: n for c, n in totals_by_class.items() if n > 0}
    if not present:
        print("No classes with samples -- nothing to analyze.")
        return

    total_images = {
        c: sum(distribution.get(c, {}).get(s, {}).get("images", 0) for s in SPLITS)
        for c in present
    }

    largest = max(present, key=present.get)
    smallest = min(present, key=present.get)
    ratio = present[largest] / present[smallest]

    print("Imbalance analysis (by annotation instances)")
    print("=" * 44)
    print(f"Largest class:  {largest} ({present[largest]} instances, {total_images[largest]} images)")
    print(f"Smallest class: {smallest} ({present[smallest]} instances, {total_images[smallest]} images)")
    print(f"Imbalance ratio (largest / smallest instances): {ratio:.1f}x")
    print()

    cutlery_instances = totals_by_class.get("cutlery", 0)
    if cutlery_instances:
        median = sorted(present.values())[len(present) // 2]
        cutlery_dominates = cutlery_instances >= 3 * median
        print(f"Cutlery instances: {cutlery_instances}; median class instances: {median}")
        print(f"Cutlery dominates heavily (>= 3x median): {'YES' if cutlery_dominates else 'no'}")
    if "counter" in MISSING_CLASSES:
        print("WARNING: class 7 'counter' has ZERO samples but stays defined in data.yaml.")
    print()

    print("Recommended strategies (NOT auto-applied -- pick before training):")
    print(f"- Cap the dominant class (e.g. random-sample {smallest}-scale per class, or a fixed cap like 2000-3000 plate images).")
    print("- Weighted sampling / class-balanced batch sampler so minority classes appear enough per epoch.")
    print("- Oversample or augment minority classes (nut_butter_jar, bowl, chopping_board).")
    print("- Deduplicate plate first (exact + cross-split duplicates) before any capping.")
    print()


def main() -> None:
    completed = [c for c in OBJECT_CLASSES if c not in MISSING_CLASSES and has_any_images(c)]
    skipped_missing = [c for c in OBJECT_CLASSES if c in MISSING_CLASSES or not has_any_images(c)]

    print("Building unified multi-class dataset")
    print("=" * 36)
    print(f"Destination: {UNIFIED_ROOT}")
    print(f"Included classes: {', '.join(completed)}")
    print(f"Excluded (missing/no data): {', '.join(skipped_missing) or 'none'}")
    print()

    reset_unified()

    distribution: dict = defaultdict(lambda: {s: {"images": 0, "instances": 0} for s in SPLITS})
    for class_name in completed:
        build_class(class_name, distribution)

    print("Verifying unified labels...")
    errors = verify_unified()
    if errors:
        print(f"VERIFICATION FAILED ({len(errors)} issue(s)):")
        for err in errors[:40]:
            print(f"  - {err}")
        raise SystemExit(1)
    print("Verification PASS: image/label pairing + class ids + coordinate ranges all OK.")
    print()

    totals_by_class = print_distribution(distribution)
    print_imbalance(distribution, totals_by_class)

    grand_images = sum(
        distribution.get(c, {}).get(s, {}).get("images", 0) for c in completed for s in SPLITS
    )
    grand_instances = sum(totals_by_class.values())
    print(f"Unified totals: {grand_images} images, {grand_instances} annotation instances "
          f"across {len(completed)} classes (counter absent).")


if __name__ == "__main__":
    main()
