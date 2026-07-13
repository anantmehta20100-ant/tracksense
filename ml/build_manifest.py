"""Build the reproducible, auditable manifest of the exact 8-class training candidate.

Scans data/training_8class_balanced as the authoritative final state, joins the
per-image provenance recorded by ml/build_balanced_training_dataset.py, and adds
SHA-256 plus source-dataset attribution.

reports/training_manifest.csv columns:
  image_path, split, source_dataset, source_image, class_name,
  canonical_class_id, model_local_class_id, sha256, augmented,
  augmentation_type, included_after_balancing
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_CLASS_TO_ID
from ml.class_schema import canonical_to_model

ROOT = Path("data/training_8class_balanced")
BUILD_MANIFEST = Path("reports/balanced_build_manifest.csv")
OUT_PATH = Path("reports/training_manifest.csv")
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

CLASS_SOURCES = {
    "nut_butter_jar": "Peanut Butter Jar 2.v1i.coco.zip; Peanut butter.v1i.coco.zip",
    "whole_nuts": "Nut classification.v1i.coco.zip",
    "hand": "Hand detection.v2i.coco.zip",
    "cutlery": "cutlery.v4i.coco.zip",
    "chopping_board": "Cutting board.v1i.coco.zip",
    "plate": "plate count.v2-plate_augmentation_without_shear.coco.zip",
    "bowl": "Bowl.v3i.coco.zip",
    # bread resolved per-file below via its source-id filename prefix.
}


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_dataset(class_name: str, filename: str, origin: str) -> str:
    if class_name != "bread":
        return CLASS_SOURCES.get(class_name, "unknown")
    probe = origin or filename
    if "breadV3i_" in probe:
        return "Bread.v3i.coco.zip"
    if "breadV1_" in probe:
        return "BREAD.v1-bread-images.coco.zip"
    return "Bread.v3i.coco.zip; BREAD.v1-bread-images.coco.zip"


def load_provenance() -> dict:
    provenance = {}
    if BUILD_MANIFEST.is_file():
        for row in csv.DictReader(BUILD_MANIFEST.open(encoding="utf-8")):
            provenance[(row["split"], row["image"])] = row
    return provenance


def main() -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"{ROOT} not found. Run ml/build_balanced_training_dataset.py first.")

    provenance = load_provenance()
    rows = []
    for split in SPLITS:
        images_dir = ROOT / split / "images"
        if not images_dir.is_dir():
            continue
        for image in sorted(images_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            class_name = image.name.split("__", 1)[0]
            canonical_id = OBJECT_CLASS_TO_ID[class_name]
            local_id = canonical_to_model(canonical_id)

            prov = provenance.get((split, image.name), {})
            source_type = prov.get("source_type", "")
            origin = prov.get("origin_image", "")
            augmented = source_type == "augmented"

            rows.append({
                "image_path": f"{split}/images/{image.name}",
                "split": split,
                "source_dataset": source_dataset(class_name, image.name, origin),
                "source_image": origin if augmented else image.name,
                "class_name": class_name,
                "canonical_class_id": canonical_id,
                "model_local_class_id": local_id,
                "sha256": sha256_of(image),
                "augmented": "yes" if augmented else "no",
                "augmentation_type": prov.get("transform", "") if augmented else "",
                "included_after_balancing": "yes",
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "split", "source_dataset", "source_image", "class_name",
                  "canonical_class_id", "model_local_class_id", "sha256", "augmented",
                  "augmentation_type", "included_after_balancing"]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_split = {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS}
    n_aug = sum(1 for r in rows if r["augmented"] == "yes")
    shas = {r["sha256"] for r in rows}
    print(f"Wrote {OUT_PATH}")
    print(f"rows: {len(rows)}  (train {by_split['train']}, valid {by_split['valid']}, test {by_split['test']})")
    print(f"augmented: {n_aug}")
    print(f"unique sha256: {len(shas)}")
    # bread must be canonical 8 / local 7 everywhere
    bread = [r for r in rows if r["class_name"] == "bread"]
    assert all(r["canonical_class_id"] == 8 and r["model_local_class_id"] == 7 for r in bread)
    print(f"bread rows: {len(bread)} -> canonical 8, model-local 7 (verified)")
    assert not any(r["class_name"] == "counter" for r in rows)
    print("counter rows: 0 (verified excluded)")


if __name__ == "__main__":
    main()
