from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path


SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-class YOLO datasets into one unified training_photos dataset."
        )
    )
    parser.add_argument(
        "dataset_roots",
        nargs="+",
        type=Path,
        help="Per-class dataset folders, each containing train/valid/test images and labels.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("ml/data.yaml"),
        help="YOLO data.yaml whose names section defines unified class IDs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/training_photos/unified"),
        help="Destination folder for the merged YOLO dataset.",
    )
    return parser.parse_args()


def clean_yaml_value(value: str) -> str:
    return value.strip().strip("'\"")


def load_class_names(data_yaml: Path) -> dict[str, int]:
    names: dict[int, str] = {}
    in_names = False

    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        stripped = line_without_comment.strip()
        if not stripped:
            continue

        if stripped == "names:":
            in_names = True
            continue

        if not in_names:
            continue

        if not raw_line.startswith((" ", "\t", "-")):
            break

        mapping_match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line_without_comment)
        if mapping_match:
            class_id = int(mapping_match.group(1))
            names[class_id] = clean_yaml_value(mapping_match.group(2))
            continue

        list_match = re.match(r"\s*-\s*(.+?)\s*$", line_without_comment)
        if list_match:
            names[len(names)] = clean_yaml_value(list_match.group(1))

    if not names:
        raise ValueError(f"No class names found in {data_yaml}")

    return {class_name: class_id for class_id, class_name in sorted(names.items())}


def infer_class_name(dataset_root: Path) -> str:
    class_name = dataset_root.name
    if class_name.endswith("_converted"):
        class_name = class_name.removesuffix("_converted")
    return class_name


def ensure_split_dirs(output_root: Path) -> None:
    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def unique_destination(destination_dir: Path, filename: str) -> Path:
    candidate = destination_dir / filename
    if not candidate.exists():
        return candidate

    source_name = Path(filename)
    for index in range(1, 100000):
        renamed = destination_dir / f"{source_name.stem}__dup{index}{source_name.suffix}"
        if not renamed.exists():
            return renamed

    raise RuntimeError(f"Could not find a safe non-colliding filename for {filename}")


def remap_label_file(source_label: Path, destination_label: Path, unified_class_id: int) -> int:
    remapped_lines: list[str] = []
    box_count = 0

    for raw_line in source_label.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 5:
            print(f"WARNING: Skipping malformed label line in {source_label}: {raw_line!r}")
            continue

        parts[0] = str(unified_class_id)
        remapped_lines.append(" ".join(parts))
        box_count += 1

    destination_label.write_text(
        "\n".join(remapped_lines) + ("\n" if remapped_lines else ""),
        encoding="utf-8",
    )
    return box_count


def merge_dataset(
    dataset_root: Path,
    output_root: Path,
    class_to_id: dict[str, int],
    summary: dict[str, dict[str, dict[str, int]]],
) -> None:
    class_name = infer_class_name(dataset_root)
    if class_name not in class_to_id:
        known_classes = ", ".join(class_to_id)
        raise ValueError(
            f"Dataset folder {dataset_root} maps to class {class_name!r}, "
            f"but that class is not in data.yaml names. Known classes: {known_classes}"
        )

    unified_class_id = class_to_id[class_name]

    for split in SPLITS:
        source_images_dir = dataset_root / split / "images"
        source_labels_dir = dataset_root / split / "labels"
        destination_images_dir = output_root / split / "images"
        destination_labels_dir = output_root / split / "labels"

        if not source_images_dir.is_dir():
            print(f"WARNING: Missing images directory: {source_images_dir}")
            continue

        for source_image in sorted(source_images_dir.iterdir()):
            if not source_image.is_file() or source_image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            destination_image = unique_destination(destination_images_dir, source_image.name)
            shutil.copy2(source_image, destination_image)
            summary[class_name][split]["images"] += 1

            source_label = source_labels_dir / f"{source_image.stem}.txt"
            if source_label.is_file():
                destination_label = destination_labels_dir / f"{destination_image.stem}.txt"
                remap_label_file(source_label, destination_label, unified_class_id)
                summary[class_name][split]["labels"] += 1
            else:
                print(f"WARNING: No label file found for image {source_image}")


def print_summary(summary: dict[str, dict[str, dict[str, int]]]) -> None:
    print("Merge summary")
    print("=" * 13)
    print("class_name | split | images | labels")
    print("-----------+-------+--------+-------")

    for class_name in sorted(summary):
        for split in SPLITS:
            counts = summary[class_name][split]
            print(
                f"{class_name} | {split} | "
                f"{counts['images']} | {counts['labels']}"
            )


def main() -> None:
    args = parse_args()
    class_to_id = load_class_names(args.data_yaml)
    ensure_split_dirs(args.output_root)

    summary: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"images": 0, "labels": 0})
    )

    for dataset_root in args.dataset_roots:
        merge_dataset(dataset_root, args.output_root, class_to_id, summary)

    print_summary(summary)


if __name__ == "__main__":
    main()
