"""Build the clean 8-class YOLO dataset from the deduplicated unified dataset.

Reads  data/training_photos/unified   (CANONICAL class ids 0..8, counter absent)
Writes data/training_8class           (MODEL-LOCAL class ids 0..7)

The only id that changes is bread: canonical 8 -> model-local 7. Every other
populated class keeps its number. `counter` (canonical 7) is excluded; if a
counter label were ever encountered this script fails loudly rather than
silently dropping or renumbering it.

The remap is applied explicitly via ml/class_schema.canonical_to_model(), which
is the single authoritative mapping. Source datasets are never modified.

Post-build verification (fails the run on any violation):
  - every image has a label and every label an image
  - every class id is in 0..7 (no canonical 8 survives, no counter)
  - coordinates numeric, centers in [0,1], w/h in (0,1]
  - no filename collisions
  - no cross-split exact (SHA-256) duplicates
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import OBJECT_ID_TO_CLASS
from ml.class_schema import (
    EXCLUDED_FROM_TRAINING,
    MODEL_LOCAL_NAMES,
    NUM_TRAINING_CLASSES,
    canonical_to_model,
    is_trained,
)

SRC_ROOT = Path("data/training_photos/unified")
DST_ROOT = Path("data/training_8class")
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def reset_dst() -> None:
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    for split in SPLITS:
        (DST_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (DST_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def remap_label(src_label: Path, dst_label: Path) -> tuple[int, Counter]:
    """Rewrite canonical ids to model-local ids. Returns (n_boxes, local_hist)."""
    out_lines: list[str] = []
    hist: Counter = Counter()
    for line_no, raw in enumerate(src_label.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise SystemExit(f"{src_label}:{line_no}: malformed label line {raw!r}")
        canonical_id = int(float(parts[0]))
        if not is_trained(canonical_id):
            name = OBJECT_ID_TO_CLASS.get(canonical_id, "?")
            raise SystemExit(
                f"{src_label}:{line_no}: canonical class {canonical_id} ({name}) is excluded "
                f"from training ({EXCLUDED_FROM_TRAINING}); refusing to drop or renumber it silently."
            )
        local_id = canonical_to_model(canonical_id)
        hist[local_id] += 1
        out_lines.append(" ".join([str(local_id), *parts[1:]]))
    if out_lines:
        dst_label.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(out_lines), hist


def build() -> dict:
    stats = {s: {"images": 0, "instances": 0} for s in SPLITS}
    hist_by_split = {s: Counter() for s in SPLITS}
    seen_names: dict[str, set] = {s: set() for s in SPLITS}

    for split in SPLITS:
        src_images = SRC_ROOT / split / "images"
        src_labels = SRC_ROOT / split / "labels"
        dst_images = DST_ROOT / split / "images"
        dst_labels = DST_ROOT / split / "labels"

        for image in sorted(src_images.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            src_label = src_labels / f"{image.stem}.txt"
            if not src_label.is_file():
                raise SystemExit(f"missing label for {image}")

            if image.name in seen_names[split]:
                raise SystemExit(f"filename collision in {split}: {image.name}")
            seen_names[split].add(image.name)

            n_boxes, hist = remap_label(src_label, dst_labels / f"{image.stem}.txt")
            if n_boxes == 0:
                continue  # never emit an image without boxes
            shutil.copy2(image, dst_images / image.name)
            stats[split]["images"] += 1
            stats[split]["instances"] += n_boxes
            hist_by_split[split] += hist

    return {"stats": stats, "hist": hist_by_split}


def verify() -> None:
    errors: list[str] = []
    by_sha: dict[str, set] = defaultdict(set)

    for split in SPLITS:
        images = {p.stem: p for p in (DST_ROOT / split / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {p.stem: p for p in (DST_ROOT / split / "labels").iterdir() if p.suffix.lower() == ".txt"}
        for stem in sorted(set(images) - set(labels)):
            errors.append(f"{split}: orphan image {stem}")
        for stem in sorted(set(labels) - set(images)):
            errors.append(f"{split}: orphan label {stem}")

        for stem, image in images.items():
            by_sha[sha256_of(image)].add(split)

        for stem, label in labels.items():
            for line_no, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not raw.strip():
                    continue
                parts = raw.split()
                cid = int(float(parts[0]))
                if not (0 <= cid < NUM_TRAINING_CLASSES):
                    errors.append(f"{label.name}:{line_no}: class id {cid} outside 0..{NUM_TRAINING_CLASSES-1}")
                cx, cy, w, h = (float(v) for v in parts[1:5])
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    errors.append(f"{label.name}:{line_no}: center out of range")
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    errors.append(f"{label.name}:{line_no}: w/h out of range")

    cross = sum(1 for splits_in in by_sha.values() if len(splits_in) > 1)
    if cross:
        errors.append(f"{cross} cross-split exact-duplicate group(s) present")

    print("Verification")
    print("-" * 12)
    print(f"cross-split exact-duplicate groups: {cross}")
    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for e in errors[:30]:
            print(f"  - {e}")
        raise SystemExit(1)
    print("PASS: pairing, class-id range 0..7, coords, collisions, cross-split leakage all OK.\n")


def main() -> None:
    print("Building clean 8-class dataset")
    print("=" * 30)
    print(f"source: {SRC_ROOT} (canonical ids)")
    print(f"dest  : {DST_ROOT} (model-local ids)")
    print(f"excluded from training: {EXCLUDED_FROM_TRAINING}")
    print(f"names: {MODEL_LOCAL_NAMES}")
    print("bread: canonical 8 -> model-local 7\n")

    reset_dst()
    result = build()
    stats, hist = result["stats"], result["hist"]

    total_img = sum(stats[s]["images"] for s in SPLITS)
    total_inst = sum(stats[s]["instances"] for s in SPLITS)
    print("Totals")
    print("-" * 6)
    for split in SPLITS:
        print(f"  {split}: images={stats[split]['images']} instances={stats[split]['instances']}")
    print(f"  TOTAL: images={total_img} instances={total_inst}\n")

    print("Per-class instances (model-local id)")
    print("-" * 36)
    print(f"{'local':>5} {'class':16} {'train':>7} {'valid':>7} {'test':>7} {'total':>8}")
    for local in range(NUM_TRAINING_CLASSES):
        t, v, te = (hist[s].get(local, 0) for s in SPLITS)
        print(f"{local:>5} {MODEL_LOCAL_NAMES[local]:16} {t:>7} {v:>7} {te:>7} {t+v+te:>8}")
    print()

    verify()


if __name__ == "__main__":
    main()
