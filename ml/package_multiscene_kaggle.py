"""Phase 9 -- Build and validate a Kaggle-safe multiscene retraining archive.

Produces tracksense_multiscene_kaggle.zip whose entries all live under a short
top-level `tracksense/` prefix, so unzipping into /kaggle/working yields
/kaggle/working/tracksense/... exactly as ml/data.8class.multiscene.kaggle.yaml
expects.

Included:
  - training code: ml/train_yolo.py, ml/class_schema.py, config/, requirements.txt
  - the multiscene data.yaml (kaggle + local)
  - the generation/validation/build scripts (reproducibility)
  - the merged dataset: data/training_8class_multiscene/{train,valid,test,stress_test}
  - the multiscene reports (json/csv) + the honesty note
  - optional local eval script (vision/test_multiscene_detection.py + deps)
  - a KAGGLE_README.md with the exact train command

Excluded (never shipped):
  - the intermediate synthetic_train/ (its content is already merged into train/)
  - any .pt weights (old best.pt, current checkpoint, backups)
  - .git, venv, __pycache__, node_modules, caches, raw zips, big legacy reports,
    the crop bank, annotated sample image folders

Then validates the zip: opens it, runs testzip() (CRC of every entry),
checks the key files are present, and reports file count, total size and the
longest internal path.

Usage:
  python ml/package_multiscene_kaggle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARC_PREFIX = "tracksense"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Some original Roboflow filenames are ~254-char base64 blobs -- at the Linux
# 255-byte NAME_MAX limit. Shorten any over-long stem deterministically; the
# SAME stem maps an image and its label to the same short name, so pairing is
# preserved (ultralytics pairs by stem, not by original filename).
MAX_STEM = 80


def short_stem(stem: str) -> str:
    if len(stem) <= MAX_STEM:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:16]
    return f"{stem[:40]}_{digest}"

# Individual files to ship (relative to repo root).
CODE_FILES = [
    "ml/train_yolo.py",
    "ml/class_schema.py",
    "ml/data.8class.multiscene.kaggle.yaml",
    "ml/data.8class.multiscene.yaml",
    "ml/create_object_crop_bank.py",
    "ml/generate_multiscene_yolo_data.py",
    "ml/validate_multiscene_dataset.py",
    "ml/build_multiscene_dataset.py",
    "config/allergens.py",
    "config/runtime_config.py",
    "vision/test_multiscene_detection.py",
    "vision/validate_yolo_checkpoint.py",
    "vision/yolo_detection_source.py",
    "requirements.txt",
]
REPORT_FILES = [
    "reports/multiscene_generation_summary.json",
    "reports/multiscene_stress_generation_summary.json",
    "reports/multiscene_class_histogram.csv",
    "reports/multiscene_cooccurrence_matrix.csv",
    "reports/multiscene_validation_report.json",
    "reports/multiscene_stress_class_histogram.csv",
    "reports/multiscene_stress_cooccurrence_matrix.csv",
    "reports/multiscene_stress_validation_report.json",
    "reports/multiscene_build_summary.json",
    "reports/multiscene_training_note.md",
]
# Dataset splits shipped in full (image + label pairs).
DATASET_ROOT = "data/training_8class_multiscene"
DATASET_SPLITS = ["train", "valid", "test", "stress_test"]

KAGGLE_README = """# TrackSense multiscene retraining (Kaggle)

This archive extracts to `/kaggle/working/tracksense/`.

## 1. Extract
```python
import zipfile
zipfile.ZipFile("/kaggle/input/<your-dataset>/tracksense_multiscene_kaggle.zip").extractall("/kaggle/working")
```

## 2. Train (8-class, multiscene)
IMPORTANT: pass BOTH `--data` and `--dataset-root` pointing at the multiscene
dataset. Without `--dataset-root`, train_yolo.py preflights its default
(single-object) dataset path, which is not shipped here.

```bash
python /kaggle/working/tracksense/ml/train_yolo.py \\
  --model yolo26n.pt \\
  --data /kaggle/working/tracksense/ml/data.8class.multiscene.kaggle.yaml \\
  --dataset-root /kaggle/working/tracksense/data/training_8class_multiscene \\
  --epochs 50 --batch 32 --imgsz 640 --device 0 --workers 4 --seed 42 \\
  --name tracksense_8class_multiscene_kaggle \\
  --project /kaggle/working/tracksense/model/checkpoints/yolo_runs
```
If batch 32 OOMs, use `--batch 16`. To skip the (slow) leakage rehash on Kaggle,
add `--skip-leakage-check` (it already passed 0 duplicates locally).

## 3. Save the weights
See ml/... post-training save cell in the handoff notes.

Schema is the exact 8-class TrackSense schema (no `counter`; `bread` = id 7).
"""


def iter_dataset_files():
    root = REPO_ROOT / DATASET_ROOT
    for split in DATASET_SPLITS:
        for leaf in ("images", "labels"):
            d = root / split / leaf
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.is_file():
                    yield p


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + validate the Kaggle multiscene retraining zip.")
    ap.add_argument("--out", default=str(REPO_ROOT / "tracksense_multiscene_kaggle.zip"))
    ap.add_argument("--no-eval-code", action="store_true", help="Omit the optional local eval script + deps.")
    args = ap.parse_args()

    out_zip = Path(args.out)
    entries: list[tuple[Path, str]] = []
    missing: list[str] = []

    code_files = list(CODE_FILES)
    if args.no_eval_code:
        code_files = [f for f in code_files if not f.startswith("vision/")]

    for rel in code_files + REPORT_FILES:
        src = REPO_ROOT / rel
        if src.is_file():
            entries.append((src, f"{ARC_PREFIX}/{rel}"))
        else:
            missing.append(rel)

    ds_count = 0
    shortened = 0
    for p in iter_dataset_files():
        rel = p.relative_to(REPO_ROOT)
        new_stem = short_stem(p.stem)
        if new_stem != p.stem:
            shortened += 1
        arc = f"{ARC_PREFIX}/{rel.parent.as_posix()}/{new_stem}{p.suffix}"
        entries.append((p, arc))
        ds_count += 1

    if missing:
        print(f"WARNING: {len(missing)} expected files missing (skipped): {missing}")

    print(f"Writing {out_zip}  ({len(entries)} entries, {ds_count} dataset files)...")
    if out_zip.exists():
        out_zip.unlink()

    total_src_bytes = 0
    max_path = ""
    with zipfile.ZipFile(out_zip, "w") as zf:
        # README first
        zf.writestr(f"{ARC_PREFIX}/KAGGLE_README.md", KAGGLE_README, compress_type=zipfile.ZIP_DEFLATED)
        for src, arc in entries:
            # store already-compressed images without recompression (fast); deflate the rest
            ctype = zipfile.ZIP_STORED if src.suffix.lower() in IMAGE_EXTS else zipfile.ZIP_DEFLATED
            zf.write(src, arcname=arc, compress_type=ctype)
            total_src_bytes += src.stat().st_size
            if len(arc) > len(max_path):
                max_path = arc

    zip_size = out_zip.stat().st_size

    # ---- validate ----
    print("Validating zip (open + testzip CRC of every entry)...")
    problems = []
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        bad = zf.testzip()  # None if all CRCs OK
        if bad is not None:
            problems.append(f"corrupt entry: {bad}")
        required = [
            f"{ARC_PREFIX}/ml/data.8class.multiscene.kaggle.yaml",
            f"{ARC_PREFIX}/ml/train_yolo.py",
            f"{ARC_PREFIX}/ml/class_schema.py",
            f"{ARC_PREFIX}/config/allergens.py",
            f"{ARC_PREFIX}/requirements.txt",
            f"{ARC_PREFIX}/KAGGLE_README.md",
        ]
        for r in required:
            if r not in names:
                problems.append(f"missing required entry: {r}")
        nameset = set(names)
        # every dataset image must have its label and vice-versa
        img_stems, lbl_stems = {}, {}
        for nm in names:
            if "/data/training_8class_multiscene/" not in nm:
                continue
            parts = nm.split("/")
            if parts[-2] == "images" and Path(nm).suffix.lower() in IMAGE_EXTS:
                img_stems.setdefault(parts[-3], set()).add(Path(nm).stem)
            elif parts[-2] == "labels" and nm.endswith(".txt"):
                lbl_stems.setdefault(parts[-3], set()).add(Path(nm).stem)
        for split in DATASET_SPLITS:
            i = img_stems.get(split, set())
            l = lbl_stems.get(split, set())
            if i != l:
                problems.append(f"{split}: {len(i - l)} images w/o label, {len(l - i)} labels w/o image")
        # ensure no forbidden content leaked in
        for nm in names:
            low = nm.lower()
            if low.endswith(".pt") or "/synthetic_train/" in low or "/.git/" in low or "__pycache__" in low or "/venv/" in low:
                problems.append(f"forbidden entry present: {nm}")

    manifest = {
        "zip": str(out_zip.as_posix()),
        "entries": len(names),
        "dataset_files": ds_count,
        "long_filenames_shortened": shortened,
        "zip_size_mb": round(zip_size / 1024 / 1024, 2),
        "uncompressed_source_mb": round(total_src_bytes / 1024 / 1024, 2),
        "longest_internal_path": max_path,
        "longest_internal_path_len": len(max_path),
        "missing_expected_files": missing,
        "valid": len(problems) == 0,
        "problems": problems[:40],
    }
    (REPO_ROOT / "reports" / "multiscene_kaggle_package_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== KAGGLE PACKAGE ===")
    print(f"zip: {out_zip}")
    print(f"entries: {len(names)}  dataset files: {ds_count}")
    print(f"zip size: {manifest['zip_size_mb']} MB  (uncompressed sources {manifest['uncompressed_source_mb']} MB)")
    print(f"longest internal path: {manifest['longest_internal_path_len']} chars")
    print(f"  {max_path}")
    print(f"manifest -> reports/multiscene_kaggle_package_manifest.json")
    if problems:
        print(f"\nZIP VALIDATION FAILED ({len(problems)} problems):")
        for p in problems[:20]:
            print("  -", p)
        raise SystemExit(1)
    print("\nZIP VALIDATION PASSED (opens, all CRCs OK, pairing intact, no forbidden content).")


if __name__ == "__main__":
    main()
