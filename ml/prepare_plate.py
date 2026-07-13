from __future__ import annotations

import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ultralytics.data.converter import convert_coco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASS_TO_ID, OBJECT_CLASSES


DOWNLOAD_DIR = Path(r"C:\Users\Anant\Downloads")
ZIP_NAME = "plate count.v2-plate_augmentation_without_shear.coco.zip"
DATASET_NAME = "plate count"
DATASET_URL = None
DATASET_ID = "plateCount"
DEST_CLASS = "plate"
DEST_CLASS_ID = OBJECT_CLASS_TO_ID[DEST_CLASS]
DEST_ROOT = Path("data/training_photos") / DEST_CLASS
TEMP_ROOT = DEST_ROOT / "_temp"
SOURCING_NOTES = Path("data/training_photos/SOURCING_NOTES.md")
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
RANDOM_SEED = 42

VALID_PLATE_NAMES = {
    "plate",
    "plates",
}

AMBIGUOUS_PLATE_NAMES = {
    "",
    "-",
    "tray",
    "trays",
    "bowl",
    "saucer",
    "saucers",
    "pan",
    "pans",
    "object",
    "item",
}


@dataclass
class JsonRecord:
    json_path: Path
    staged_name: str
    data: dict
    inferred_split: str | None


@dataclass
class ImageRecord:
    json_record: JsonRecord
    image: dict
    split: str | None


def require_zip_file() -> Path:
    zip_path = DOWNLOAD_DIR / ZIP_NAME
    if not zip_path.is_file():
        print(f"Missing required source zip file: {zip_path}")
        print("Download the Roboflow COCO zip to C:\\Users\\Anant\\Downloads\\ and rerun.")
        raise SystemExit(1)

    print(f"Found required source zip file: {zip_path}")
    print()
    return zip_path


def reset_temp_root() -> None:
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def ensure_destination_dirs() -> None:
    for split in SPLITS:
        for leaf in ("images", "labels"):
            path = DEST_ROOT / split / leaf
            path.mkdir(parents=True, exist_ok=True)
            gitkeep = path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.write_text("\n", encoding="utf-8")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_dir)


def is_coco_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(key in data for key in ("images", "annotations", "categories"))


def infer_split(path: Path, dataset_root: Path) -> str | None:
    try:
        parts = [part.lower() for part in path.relative_to(dataset_root).parts]
    except ValueError:
        parts = [part.lower() for part in path.parts]

    for part in parts:
        clean = part.replace("_", "-")
        if clean == "train":
            return "train"
        if clean in {"valid", "val", "validation"}:
            return "valid"
        if clean == "test":
            return "test"
    return None


def find_coco_jsons(extract_dir: Path) -> list[JsonRecord]:
    records: list[JsonRecord] = []
    json_paths = sorted(path for path in extract_dir.rglob("*.json") if is_coco_json(path))

    for index, json_path in enumerate(json_paths):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        split = infer_split(json_path, extract_dir)
        split_name = split or "unsplit"
        records.append(
            JsonRecord(
                json_path=json_path,
                staged_name=f"{DATASET_ID}_{split_name}_{index}",
                data=data,
                inferred_split=split,
            )
        )

    if not records:
        print(f"No COCO annotation JSON files found in {extract_dir}")
        raise SystemExit(1)

    return records


def normalize_class_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def classify_category(name: str) -> tuple[bool, str]:
    normalized = normalize_class_name(name)
    raw = name.strip().lower()

    if normalized in VALID_PLATE_NAMES:
        return True, f"{name!r} is a clearly valid plate category."

    if raw in AMBIGUOUS_PLATE_NAMES or normalized in AMBIGUOUS_PLATE_NAMES:
        return False, f"{name!r} is ambiguous or a different object type (tray/bowl/saucer/pan), not plate."

    return False, f"{name!r} is not clearly a plate annotation label."


def clear_destination_files() -> None:
    removed = 0
    for split in SPLITS:
        for leaf in ("images", "labels"):
            folder = DEST_ROOT / split / leaf
            for path in folder.iterdir():
                if path.is_file() and path.name != ".gitkeep":
                    path.unlink()
                    removed += 1
    print(f"Cleared {removed} existing plate image/label file(s), preserving .gitkeep placeholders.")
    print()


def validate_destination_cleared() -> None:
    stale_files: list[Path] = []
    for split in SPLITS:
        for leaf in ("images", "labels"):
            folder = DEST_ROOT / split / leaf
            stale_files.extend(path for path in folder.iterdir() if path.is_file() and path.name != ".gitkeep")

    if stale_files:
        print("ERROR: Stale files remain after destination clear:")
        for path in stale_files[:20]:
            print(f"- {path}")
        raise RuntimeError("Destination clear validation failed.")

    print("Stale-file validation: destination image/label folders are clear before merge.")
    print()


def summarize_records(
    records: list[JsonRecord],
) -> tuple[set[tuple[str, int]], dict[str, str], dict[str, str], dict[str, str]]:
    accepted_category_keys: set[tuple[str, int]] = set()
    accepted_reasons: dict[str, str] = {}
    zero_instance_valid_reasons: dict[str, str] = {}
    excluded_reasons: dict[str, str] = {}

    print("COCO inspection summary")
    print("=" * 23)
    for record in records:
        categories = record.data.get("categories", [])
        annotations = record.data.get("annotations", [])
        images = record.data.get("images", [])
        category_counter = Counter(ann.get("category_id") for ann in annotations)

        print(f"Dataset: {DATASET_NAME} ({DATASET_ID})")
        print(f"Annotation JSON: {record.json_path}")
        print(f"Source split: {record.inferred_split or 'not provided'}")
        print(f"Images: {len(images)}")
        print(f"Annotation instances: {len(annotations)}")
        print("Categories:")

        for category in categories:
            category_id = int(category["id"])
            category_name = str(category.get("name", ""))
            instance_count = category_counter.get(category_id, 0)
            accepted, reason = classify_category(category_name)
            if accepted and instance_count > 0:
                decision = f"accepted -> {DEST_CLASS}"
            elif accepted:
                decision = "valid plate label, but zero instances -> no labels copied (Roboflow root/supercategory placeholder)"
            elif instance_count == 0:
                decision = "excluded zero-instance placeholder -> no annotations reference it, no manual review needed"
            else:
                decision = "excluded -> needs manual review (ambiguous category WITH instances; its annotations are dropped)"
            print(f"- id={category_id}, name={category_name!r}, instances={instance_count}: {decision}")
            print(f"  reasoning: {reason}")

            category_key = f"{DATASET_NAME}: id={category_id}, name={category_name}"
            if accepted and instance_count > 0:
                accepted_category_keys.add((record.staged_name, category_id))
                accepted_reasons[category_key] = reason
            elif accepted:
                zero_instance_valid_reasons[category_key] = reason
            elif instance_count == 0:
                # Non-target category with no annotations (e.g. Roboflow root
                # placeholder named after the dataset slug). Harmless -- record
                # it informationally but do NOT force "needs review".
                zero_instance_valid_reasons[category_key] = (
                    f"{reason} Zero instances, treated as an excludable placeholder."
                )
            else:
                excluded_reasons[category_key] = reason

        if not categories:
            print("- No categories found.")
        print()

    if excluded_reasons:
        print("WARNING: Ambiguous or non-plate classes were found and will be excluded:")
        for category_key, reason in excluded_reasons.items():
            print(f"- {category_key}: {reason}")
        print()

    return accepted_category_keys, accepted_reasons, zero_instance_valid_reasons, excluded_reasons


def assign_splits(records: list[JsonRecord]) -> tuple[list[ImageRecord], bool]:
    image_records: list[ImageRecord] = []

    for record in records:
        for image in record.data.get("images", []):
            image_record = ImageRecord(json_record=record, image=image, split=record.inferred_split)
            image_records.append(image_record)

    # A source that provides only a train split (or no split at all) has no
    # real valid/test split to preserve. Leaving it as-is would give this
    # class zero validation/test coverage, so force a deterministic seeded
    # 80/10/10 split over ALL images. If the source already carries valid or
    # test images, only fill in the images that are missing a split.
    present_splits = {r.split for r in image_records if r.split is not None}
    has_real_split = bool(present_splits & {"valid", "test"})

    if has_real_split:
        to_split = [r for r in image_records if r.split is None]
    else:
        to_split = list(image_records)

    if not to_split:
        return image_records, False

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(to_split)
    total = len(to_split)
    train_cutoff = int(total * 0.8)
    valid_cutoff = train_cutoff + int(total * 0.1)

    for index, image_record in enumerate(to_split):
        if index < train_cutoff:
            image_record.split = "train"
        elif index < valid_cutoff:
            image_record.split = "valid"
        else:
            image_record.split = "test"

    return image_records, True


def stage_jsons_for_conversion(records: list[JsonRecord], staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        staged_path = staging_dir / f"{record.staged_name}.json"
        staged_path.write_text(json.dumps(record.data), encoding="utf-8")


def run_ultralytics_conversion(staging_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    print("Running ultralytics.data.converter.convert_coco(..., cls91to80=False)")
    convert_coco(labels_dir=str(staging_dir), save_dir=str(output_dir), cls91to80=False)
    print()


def image_index(extract_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in extract_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.name].append(path)
    return index


def find_image_file(image: dict, record: JsonRecord, extract_dir: Path, indexed_images: dict[str, list[Path]]) -> Path | None:
    file_name = str(image.get("file_name", ""))
    candidates = [
        record.json_path.parent / file_name,
        extract_dir / file_name,
        extract_dir / (record.inferred_split or "") / file_name,
        record.json_path.parent / Path(file_name).name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    basename_matches = indexed_images.get(Path(file_name).name, [])
    if len(basename_matches) == 1:
        return basename_matches[0]

    if len(basename_matches) > 1:
        same_split = [
            path
            for path in basename_matches
            if record.inferred_split and record.inferred_split in [part.lower() for part in path.parts]
        ]
        if len(same_split) == 1:
            return same_split[0]

    return None


def label_path_for(image: dict, record: JsonRecord, conversion_dir: Path) -> Path | None:
    file_name = Path(str(image.get("file_name", ""))).with_suffix(".txt")
    label_root = conversion_dir / "labels" / record.staged_name
    direct_path = label_root / file_name
    if direct_path.is_file():
        return direct_path

    basename_matches = list(label_root.rglob(file_name.name)) if label_root.is_dir() else []
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None


def accepted_annotations_by_image(
    records: list[JsonRecord],
    accepted_category_keys: set[tuple[str, int]],
) -> dict[tuple[str, int], list[dict]]:
    result: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        for ann in record.data.get("annotations", []):
            category_id = int(ann.get("category_id"))
            if (record.staged_name, category_id) in accepted_category_keys and not ann.get("iscrowd", False):
                result[(record.staged_name, int(ann["image_id"]))].append(ann)
    return result


def report_source_filename_duplicates(image_records: list[ImageRecord]) -> list[str]:
    exact: dict[str, list[str]] = defaultdict(list)
    normalized: dict[str, list[str]] = defaultdict(list)

    for image_record in image_records:
        filename = Path(str(image_record.image.get("file_name", ""))).name
        descriptor = f"{image_record.split}/{filename}"
        exact[filename.lower()].append(descriptor)
        normalized_key = "".join(ch for ch in Path(filename).stem.lower() if ch.isalnum())
        normalized[normalized_key].append(descriptor)

    warnings: list[str] = []
    exact_duplicates = {name: items for name, items in exact.items() if len(items) > 1}
    near_duplicates = {
        name: items
        for name, items in normalized.items()
        if name and len(items) > 1 and len({Path(item).name.lower() for item in items}) > 1
    }

    print("Source filename duplicate check")
    print("=" * 31)
    if exact_duplicates:
        print(f"Exact duplicate source filenames detected: {len(exact_duplicates)} group(s) (showing up to 10).")
        for items in list(exact_duplicates.values())[:10]:
            print(f"- {', '.join(items)}")
            warnings.append(f"Exact duplicate source filenames: {', '.join(items)}")
    else:
        print("No exact duplicate source filenames detected.")

    if near_duplicates:
        print(f"Near-identical source filename stems detected: {len(near_duplicates)} group(s) (showing up to 10).")
        for items in list(near_duplicates.values())[:10]:
            print(f"- {', '.join(items)}")
            warnings.append(f"Near-identical source filename stems: {', '.join(items)}")
    else:
        print("No near-identical source filename stems detected.")
    print()
    return warnings


def report_zero_accepted_source_images(
    image_records: list[ImageRecord],
    accepted_by_image: dict[tuple[str, int], list[dict]],
) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for image_record in image_records:
        image_id = int(image_record.image["id"])
        key = (image_record.json_record.staged_name, image_id)
        if not accepted_by_image.get(key):
            grouped[image_record.split or "unknown"].append(str(image_record.image.get("file_name", "")))

    print("Source images with zero accepted annotations")
    print("=" * 42)
    warnings: list[str] = []
    if not grouped:
        print("None.")
        print()
        return warnings

    for split, filenames in sorted(grouped.items()):
        message = f"{DATASET_NAME} / {split}: {len(filenames)} image(s)"
        warnings.append(f"Zero accepted source annotations: {message}")
        print(message)
        for filename in filenames[:10]:
            print(f"- {filename}")
        if len(filenames) > 10:
            print(f"- ... {len(filenames) - 10} more")
    print()
    return warnings


def unique_destination(split: str, source_image: Path, collision_reports: list[str]) -> Path:
    destination_dir = DEST_ROOT / split / "images"
    candidate = destination_dir / source_image.name
    if not candidate.exists():
        return candidate

    prefixed = destination_dir / f"{DATASET_ID}_{source_image.name}"
    if not prefixed.exists():
        collision_reports.append(f"{split}: {source_image.name} -> {prefixed.name}")
        return prefixed

    for index in range(1, 100000):
        renamed = destination_dir / f"{DATASET_ID}_{source_image.stem}__dup{index}{source_image.suffix}"
        if not renamed.exists():
            collision_reports.append(f"{split}: {source_image.name} -> {renamed.name}")
            return renamed

    raise RuntimeError(f"Could not create non-colliding destination for {source_image}")


def remap_label(source_label: Path, destination_label: Path, accepted_yolo_ids: set[int]) -> int:
    remapped: list[str] = []

    for raw_line in source_label.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            yolo_class_id = int(float(parts[0]))
        except (IndexError, ValueError):
            continue
        if yolo_class_id not in accepted_yolo_ids:
            continue
        parts[0] = str(DEST_CLASS_ID)
        remapped.append(" ".join(parts))

    if remapped:
        destination_label.write_text("\n".join(remapped) + "\n", encoding="utf-8")
    return len(remapped)


def merge_into_destination(
    image_records: list[ImageRecord],
    accepted_by_image: dict[tuple[str, int], list[dict]],
    accepted_category_keys: set[tuple[str, int]],
    extract_dir: Path,
    conversion_dir: Path,
    collision_reports: list[str],
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"images": 0, "labels": 0, "instances": 0})
    indexed_images = image_index(extract_dir)
    accepted_yolo_ids_by_record: dict[str, set[int]] = defaultdict(set)
    for staged_name, category_id in accepted_category_keys:
        accepted_yolo_ids_by_record[staged_name].add(category_id - 1)

    for image_record in image_records:
        image_id = int(image_record.image["id"])
        accepted_annotations = accepted_by_image.get((image_record.json_record.staged_name, image_id), [])
        if not accepted_annotations:
            continue

        split = image_record.split
        if split not in SPLITS:
            raise RuntimeError(f"Image {image_record.image.get('file_name')} has no valid split.")

        source_image = find_image_file(image_record.image, image_record.json_record, extract_dir, indexed_images)
        if source_image is None:
            print(f"WARNING: Could not locate image file {image_record.image.get('file_name')}; skipping.")
            continue

        source_label = label_path_for(image_record.image, image_record.json_record, conversion_dir)
        if source_label is None:
            print(f"WARNING: Could not locate converted label for {image_record.image.get('file_name')}; skipping.")
            continue

        destination_image = unique_destination(split, source_image, collision_reports)
        destination_label = DEST_ROOT / split / "labels" / f"{destination_image.stem}.txt"

        accepted_yolo_ids = accepted_yolo_ids_by_record[image_record.json_record.staged_name]
        copied_instances = remap_label(source_label, destination_label, accepted_yolo_ids)
        if copied_instances == 0:
            continue

        shutil.copy2(source_image, destination_image)
        summary[split]["images"] += 1
        summary[split]["labels"] += 1
        summary[split]["instances"] += copied_instances

    return summary


def count_final_files() -> dict[str, dict[str, int]]:
    final_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        images = [
            path
            for path in (DEST_ROOT / split / "images").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        labels = [
            path
            for path in (DEST_ROOT / split / "labels").iterdir()
            if path.is_file() and path.name != ".gitkeep" and path.suffix.lower() == ".txt"
        ]
        final_counts[split] = {"images": len(images), "labels": len(labels)}
    return final_counts


def validate_final_dataset() -> tuple[dict[str, dict[str, int]], int]:
    final_counts: dict[str, dict[str, int]] = {}
    errors: list[str] = []
    total_instances = 0

    for split in SPLITS:
        images_dir = DEST_ROOT / split / "images"
        labels_dir = DEST_ROOT / split / "labels"
        images = {
            path.stem: path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        labels = {
            path.stem: path
            for path in labels_dir.iterdir()
            if path.is_file() and path.name != ".gitkeep" and path.suffix.lower() == ".txt"
        }

        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        if missing_labels:
            errors.append(f"{split}: {len(missing_labels)} image(s) missing matching label file.")
        if missing_images:
            errors.append(f"{split}: {len(missing_images)} label file(s) missing matching image.")

        split_instances = 0
        for label_path in labels.values():
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                try:
                    class_id = int(float(parts[0]))
                except (IndexError, ValueError):
                    errors.append(f"{label_path}:{line_number}: could not parse YOLO class id.")
                    continue
                if class_id != DEST_CLASS_ID:
                    errors.append(f"{label_path}:{line_number}: expected class id {DEST_CLASS_ID}, found {class_id}.")
                split_instances += 1

        final_counts[split] = {"images": len(images), "labels": len(labels), "instances": split_instances}
        total_instances += split_instances

    print("Final validation")
    print("=" * 16)
    for split in SPLITS:
        counts = final_counts[split]
        print(f"{split}: images={counts['images']}, labels={counts['labels']}, instances={counts['instances']}")

    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"- {error}")
        raise RuntimeError("Final dataset validation failed.")

    print("Image/label pairing validation: PASS")
    print(f"YOLO class id validation: PASS - every annotation class id is {DEST_CLASS_ID}")
    print(f"Total final annotation instances: {total_instances}")
    print()
    return final_counts, total_instances


def update_sourcing_notes(
    final_total_images: int,
    final_total_instances: int,
    excluded_reasons: dict[str, str],
    zero_instance_valid_reasons: dict[str, str],
) -> str:
    status = "needs review" if excluded_reasons else "complete"
    source = ZIP_NAME
    if excluded_reasons:
        excluded = "; ".join(excluded_reasons)
        notes = (
            f"Combined {final_total_images} accepted images / {final_total_instances} annotations; "
            f"excluded or ambiguous classes need manual review: {excluded}."
        )
    else:
        zero_note = ""
        if zero_instance_valid_reasons:
            zero_note = " Zero-instance Roboflow root/supercategory placeholder excluded (no annotations reference it)."
        notes = (
            f"Combined {final_total_images} images / {final_total_instances} annotations "
            f"(single heavily-augmented source, no source valid/test split -- auto 80/10/10 split applied); "
            f"remapped to global class id {DEST_CLASS_ID}.{zero_note} "
            "NOTE: this is by far the largest per-class dataset in the project -- flag for imbalance review before training."
        )
    new_row = f"| plate | {status} | {source} | {notes} |"

    lines = SOURCING_NOTES.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("| plate |"):
            updated.append(new_row)
            replaced = True
        else:
            updated.append(line)

    if not replaced:
        updated.append(new_row)

    SOURCING_NOTES.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return new_row


def print_merge_summary(summary: dict[str, dict[str, int]], final_counts: dict[str, dict[str, int]]) -> None:
    print("Merge summary")
    print("=" * 13)
    total_images = 0
    total_labels = 0
    total_instances = 0
    for split in SPLITS:
        split_summary = summary[split]
        final = final_counts[split]
        total_images += split_summary["images"]
        total_labels += split_summary["labels"]
        total_instances += split_summary["instances"]
        match_text = "MATCH" if final["images"] == final["labels"] else "MISMATCH"
        print(
            f"{split}: copied_images={split_summary['images']}, "
            f"copied_labels={split_summary['labels']}, "
            f"copied_instances={split_summary['instances']}, "
            f"final_images={final['images']}, final_labels={final['labels']} ({match_text})"
        )

    print(f"TOTAL copied from source dataset: images={total_images}, labels={total_labels}, instances={total_instances}")
    print()


def print_collision_summary(collision_reports: list[str]) -> list[str]:
    print("Destination filename collision check")
    print("=" * 36)
    if not collision_reports:
        print("No destination filename collisions detected; no copied file overwrote another.")
        print()
        return []

    print(f"Collisions safely renamed: {len(collision_reports)} (showing up to 20).")
    for report in collision_reports[:20]:
        print(f"- {report}")
    print()
    return [f"Destination collision safely renamed: {report}" for report in collision_reports]


def print_class_order() -> None:
    print("Canonical TrackSense class order")
    print("=" * 33)
    for index, class_name in enumerate(OBJECT_CLASSES):
        print(f"{index}: {class_name}")
    print(f"{DEST_CLASS} global class id: {DEST_CLASS_ID}")
    print()


def main() -> None:
    print_class_order()
    zip_path = require_zip_file()
    reset_temp_root()
    ensure_destination_dirs()

    extract_dir = TEMP_ROOT / "source"
    print(f"Extracting {zip_path.name} -> {extract_dir}")
    extract_zip(zip_path, extract_dir)
    records = find_coco_jsons(extract_dir)
    print()

    accepted_category_keys, accepted_reasons, zero_instance_valid_reasons, excluded_reasons = summarize_records(records)
    if not accepted_category_keys:
        print("No acceptable plate categories with annotations found. Stopping without merge.")
        raise SystemExit(1)

    image_records, automatic_split_applied = assign_splits(records)
    if automatic_split_applied:
        print(f"Automatic 80/10/10 split applied to unsplit images with random seed {RANDOM_SEED}.")
    else:
        print("Source train/valid/test splits were detected; no automatic split was needed.")
    print()

    anomaly_reports = report_source_filename_duplicates(image_records)

    print("Accepted class mapping decisions:")
    for category_key, reason in accepted_reasons.items():
        print(f"- {category_key} -> class {DEST_CLASS_ID} {DEST_CLASS} ({reason})")
    for category_key, reason in zero_instance_valid_reasons.items():
        print(f"- {category_key} -> valid plate label, zero instances copied ({reason})")
    print()

    staging_dir = TEMP_ROOT / "annotations_for_convert"
    conversion_dir = TEMP_ROOT / "converted_yolo"
    stage_jsons_for_conversion(records, staging_dir)
    run_ultralytics_conversion(staging_dir, conversion_dir)

    clear_destination_files()
    validate_destination_cleared()
    accepted_by_image = accepted_annotations_by_image(records, accepted_category_keys)
    anomaly_reports.extend(report_zero_accepted_source_images(image_records, accepted_by_image))

    collision_reports: list[str] = []
    copied_summary = merge_into_destination(
        image_records,
        accepted_by_image,
        accepted_category_keys,
        extract_dir,
        conversion_dir,
        collision_reports,
    )
    final_counts = count_final_files()
    print_merge_summary(copied_summary, final_counts)
    anomaly_reports.extend(print_collision_summary(collision_reports))
    validated_counts, final_total_instances = validate_final_dataset()

    final_total_images = sum(validated_counts[split]["images"] for split in SPLITS)
    notes_row = update_sourcing_notes(
        final_total_images,
        final_total_instances,
        excluded_reasons,
        zero_instance_valid_reasons,
    )
    print("Updated SOURCING_NOTES.md plate row:")
    print(notes_row)
    print()

    print("Warnings / anomalies")
    print("=" * 20)
    if anomaly_reports:
        for report in anomaly_reports:
            print(f"- {report}")
    else:
        print("None.")
    print()

    shutil.rmtree(TEMP_ROOT)
    print(f"Deleted temporary working folder: {TEMP_ROOT}")


if __name__ == "__main__":
    main()
