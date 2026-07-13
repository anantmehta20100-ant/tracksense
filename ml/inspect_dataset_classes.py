from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DATA_ROOT = "data/training_photos/cutlery_converted"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
METADATA_READMES = {"readme.dataset.txt", "readme.roboflow.txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect YOLO label files to determine dataset class IDs."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Converted YOLO dataset root. Default: {DEFAULT_DATA_ROOT}",
    )
    return parser.parse_args()


def parse_class_id(raw_value: str, label_path: Path, line_number: int) -> int | None:
    try:
        return int(raw_value)
    except ValueError:
        try:
            float_value = float(raw_value)
        except ValueError:
            print(
                f"WARNING: Could not parse class ID in {label_path} "
                f"line {line_number}: {raw_value!r}"
            )
            return None

        if float_value.is_integer():
            return int(float_value)

        print(
            f"WARNING: Non-integer class ID in {label_path} "
            f"line {line_number}: {raw_value!r}"
        )
        return None


def read_label_file(label_path: Path) -> tuple[Counter[int], set[int]]:
    instance_counts: Counter[int] = Counter()
    image_class_ids: set[int] = set()

    try:
        lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        print(f"WARNING: Could not read label file {label_path}: {exc}")
        return instance_counts, image_class_ids

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 5:
            print(
                f"WARNING: Expected at least 5 YOLO fields in {label_path} "
                f"line {line_number}, got {len(parts)}: {stripped!r}"
            )

        class_id = parse_class_id(parts[0], label_path, line_number)
        if class_id is None:
            continue

        instance_counts[class_id] += 1
        image_class_ids.add(class_id)

    return instance_counts, image_class_ids


def count_split_images(data_root: Path, split: str, label_files: list[Path]) -> tuple[int, str]:
    images_dir = data_root / split / "images"
    if images_dir.is_dir():
        image_count = sum(
            1
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        return image_count, f"{split}/images"

    return len(label_files), f"{split}/labels fallback"


def inspect_labels(data_root: Path) -> dict[str, dict[str, object]]:
    split_stats: dict[str, dict[str, object]] = {}

    for split in SPLITS:
        labels_dir = data_root / split / "labels"
        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []

        split_instance_counts: Counter[int] = Counter()
        split_image_sets: dict[int, set[str]] = defaultdict(set)

        for label_path in label_files:
            file_counts, file_class_ids = read_label_file(label_path)
            split_instance_counts.update(file_counts)

            image_key = label_path.stem
            for class_id in file_class_ids:
                split_image_sets[class_id].add(image_key)

        image_count, denominator_source = count_split_images(data_root, split, label_files)

        split_stats[split] = {
            "labels_dir": labels_dir,
            "label_files": label_files,
            "image_count": image_count,
            "denominator_source": denominator_source,
            "instance_counts": split_instance_counts,
            "image_sets": split_image_sets,
        }

    return split_stats


def find_metadata_files(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []

    matches: list[Path] = []
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue

        lower_name = path.name.lower()
        if lower_name in METADATA_READMES or path.suffix.lower() == ".yaml":
            matches.append(path)

    return sorted(matches)


def print_metadata_files(data_root: Path) -> None:
    print("Metadata files found under data root")
    print("=" * 38)

    metadata_files = find_metadata_files(data_root)
    if not metadata_files:
        print("No README.dataset.txt, README.roboflow.txt, or *.yaml files found.")
        print()
        return

    for path in metadata_files:
        try:
            relative_path = path.relative_to(data_root)
        except ValueError:
            relative_path = path

        print(f"--- {relative_path} ---")
        try:
            contents = path.read_text(encoding="utf-8", errors="replace").rstrip()
        except OSError as exc:
            print(f"[Could not read file: {exc}]")
        else:
            print(contents if contents else "[empty file]")
        print()


def format_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.2f}%"


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in string_rows))
        for index, header in enumerate(headers)
    ]

    header_line = " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator_line = "-+-".join("-" * width for width in widths)

    print(header_line)
    print(separator_line)
    for row in string_rows:
        print(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def build_combined_counts(
    split_stats: dict[str, dict[str, object]],
) -> tuple[Counter[int], dict[int, set[str]]]:
    total_instances: Counter[int] = Counter()
    combined_image_sets: dict[int, set[str]] = defaultdict(set)

    for split, stats in split_stats.items():
        instance_counts = stats["instance_counts"]
        image_sets = stats["image_sets"]

        if isinstance(instance_counts, Counter):
            total_instances.update(instance_counts)

        if isinstance(image_sets, dict):
            for class_id, image_keys in image_sets.items():
                combined_image_sets[class_id].update(
                    f"{split}/{image_key}" for image_key in image_keys
                )

    return total_instances, combined_image_sets


def print_summary(data_root: Path, split_stats: dict[str, dict[str, object]]) -> None:
    print("Dataset root")
    print("=" * 12)
    print(data_root)
    print()

    print("Split inventory")
    print("=" * 15)
    inventory_rows: list[list[object]] = []
    for split in SPLITS:
        stats = split_stats[split]
        labels_dir = stats["labels_dir"]
        label_files = stats["label_files"]
        labels_dir_status = "found" if isinstance(labels_dir, Path) and labels_dir.is_dir() else "missing"
        inventory_rows.append(
            [
                split,
                labels_dir_status,
                len(label_files) if isinstance(label_files, list) else 0,
                stats["image_count"],
                stats["denominator_source"],
            ]
        )
    print_table(
        [
            "split",
            "labels_dir",
            "label_files",
            "image_count",
            "percent_denominator",
        ],
        inventory_rows,
    )
    print()

    total_instances, combined_image_sets = build_combined_counts(split_stats)
    class_ids = sorted(total_instances)

    print("Class ID summary")
    print("=" * 16)
    if not class_ids:
        print("No class IDs found in train/valid/test label files.")
        print()
        return

    print(f"Unique class IDs overall: {len(class_ids)}")
    print(f"Class IDs found: {', '.join(str(class_id) for class_id in class_ids)}")
    print()

    summary_rows: list[list[object]] = []
    for class_id in class_ids:
        row: list[object] = [
            class_id,
            total_instances[class_id],
            len(combined_image_sets[class_id]),
        ]

        for split in SPLITS:
            stats = split_stats[split]
            image_sets = stats["image_sets"]
            denominator = stats["image_count"]
            split_image_count = (
                len(image_sets.get(class_id, set())) if isinstance(image_sets, dict) else 0
            )
            row.append(
                format_percent(
                    split_image_count,
                    denominator if isinstance(denominator, int) else 0,
                )
            )

        summary_rows.append(row)

    print_table(
        [
            "class_id",
            "total_instances",
            "images_containing_it",
            "percent_of_train_images",
            "percent_of_valid_images",
            "percent_of_test_images",
        ],
        summary_rows,
    )
    print()

    print("Per-split counts")
    print("=" * 16)
    split_count_rows: list[list[object]] = []
    for class_id in class_ids:
        row = [class_id]
        for split in SPLITS:
            stats = split_stats[split]
            instance_counts = stats["instance_counts"]
            image_sets = stats["image_sets"]
            row.extend(
                [
                    instance_counts[class_id] if isinstance(instance_counts, Counter) else 0,
                    len(image_sets.get(class_id, set())) if isinstance(image_sets, dict) else 0,
                ]
            )
        row.extend([total_instances[class_id], len(combined_image_sets[class_id])])
        split_count_rows.append(row)

    print_table(
        [
            "class_id",
            "train_instances",
            "train_images",
            "valid_instances",
            "valid_images",
            "test_instances",
            "test_images",
            "all_instances",
            "all_images",
        ],
        split_count_rows,
    )
    print()

    if class_ids == [0]:
        print("=" * 78)
        print("SINGLE-CLASS DATASET DETECTED: only class ID 0 appears in all labels.")
        print(
            "Separate knife/fork/spoon distinction is NOT available from this "
            "YOLO label set."
        )
        print("=" * 78)
        print()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    print_metadata_files(data_root)
    split_stats = inspect_labels(data_root)
    print_summary(data_root, split_stats)


if __name__ == "__main__":
    main()
