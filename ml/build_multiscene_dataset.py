"""Phase 7 -- Assemble the merged multiscene retraining dataset.

Builds data/training_8class_multiscene/ in the exact layout train_yolo.py
expects (root/{split}/{images,labels}) by combining:

  train  = original balanced TRAIN  +  synthetic multi-object scenes
  valid  = original balanced VALID  (unchanged)
  test   = original balanced TEST   (unchanged)

The original single-object images are preserved so the model keeps its strong
isolated-object performance; the synthetic scenes add the missing co-occurrence
signal. Validation and test splits are copied verbatim and NEVER receive
synthetic scenes, so reported mAP stays comparable to the old model.

Images are hard-linked by default (same NTFS volume -> ~no extra disk, real
files for zipping and for ultralytics), with an automatic copy fallback. Label
files are always copied (tiny).

The held-out stress_test/ set is produced separately by
ml/generate_multiscene_yolo_data.py --stress and is intentionally NOT part of
train/valid/test.

Usage:
  python ml/build_multiscene_dataset.py --mode hardlink
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    if dst.exists():
        return "exists"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy-fallback"
    shutil.copy2(src, dst)
    return "copy"


def transfer_split(src_root: Path, dst_root: Path, split: str, mode: str, image_mode: str) -> dict:
    src_img = src_root / split / "images"
    src_lbl = src_root / split / "labels"
    dst_img = dst_root / split / "images"
    dst_lbl = dst_root / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    for img in sorted(src_img.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = src_lbl / f"{img.stem}.txt"
        if not lbl.is_file():
            counts["orphan_image_skipped"] += 1
            continue
        link_or_copy(img, dst_img / img.name, image_mode)
        # labels are always copied (they are tiny and we never want to mutate source)
        if not (dst_lbl / lbl.name).exists():
            shutil.copy2(lbl, dst_lbl / lbl.name)
        counts["pairs"] += 1
    return dict(counts)


def transfer_flat(src_dir: Path, dst_root: Path, split: str, image_mode: str) -> dict:
    """Transfer a flat images/ + labels/ dir (the synthetic set) into a split."""
    src_img = src_dir / "images"
    src_lbl = src_dir / "labels"
    dst_img = dst_root / split / "images"
    dst_lbl = dst_root / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for img in sorted(src_img.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = src_lbl / f"{img.stem}.txt"
        if not lbl.is_file():
            counts["orphan_image_skipped"] += 1
            continue
        link_or_copy(img, dst_img / img.name, image_mode)
        if not (dst_lbl / lbl.name).exists():
            shutil.copy2(lbl, dst_lbl / lbl.name)
        counts["pairs"] += 1
    return dict(counts)


def verify_split(dst_root: Path, split: str) -> dict:
    img_dir = dst_root / split / "images"
    lbl_dir = dst_root / split / "labels"
    imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    lbls = {p.stem for p in lbl_dir.glob("*.txt")}
    return {
        "images": len(imgs),
        "labels": len(lbls),
        "unpaired_images": len(imgs - lbls),
        "unpaired_labels": len(lbls - imgs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the merged multiscene retraining dataset.")
    ap.add_argument("--original-root", default=str(REPO_ROOT / "data" / "training_8class_balanced"))
    ap.add_argument("--synthetic", default=str(REPO_ROOT / "data" / "training_8class_multiscene" / "synthetic_train"))
    ap.add_argument("--out-root", default=str(REPO_ROOT / "data" / "training_8class_multiscene"))
    ap.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink",
                    help="How to transfer image files (labels are always copied).")
    args = ap.parse_args()

    src_root = Path(args.original_root)
    syn = Path(args.synthetic)
    dst_root = Path(args.out_root)
    if not src_root.is_dir():
        raise SystemExit(f"FAIL: original root not found: {src_root}")
    if not (syn / "images").is_dir():
        raise SystemExit(f"FAIL: synthetic images not found: {syn/'images'}. Run the generator first.")

    print(f"Building {dst_root}  (image transfer mode: {args.mode})")
    result = {"mode": args.mode, "original_root": str(src_root.as_posix()),
              "synthetic": str(syn.as_posix()), "out_root": str(dst_root.as_posix()), "splits": {}}

    for split in ("train", "valid", "test"):
        c = transfer_split(src_root, dst_root, split, args.mode, args.mode)
        result["splits"][split] = {"from_original": c}
        print(f"  {split}: original pairs -> {c}")

    csyn = transfer_flat(syn, dst_root, "train", args.mode)
    result["splits"]["train"]["from_synthetic"] = csyn
    print(f"  train: synthetic pairs -> {csyn}")

    print("\nVerifying merged splits...")
    all_ok = True
    for split in ("train", "valid", "test"):
        v = verify_split(dst_root, split)
        result["splits"][split]["verified"] = v
        ok = v["unpaired_images"] == 0 and v["unpaired_labels"] == 0
        all_ok = all_ok and ok
        print(f"  {split}: images={v['images']} labels={v['labels']} "
              f"unpaired_img={v['unpaired_images']} unpaired_lbl={v['unpaired_labels']} {'OK' if ok else 'FAIL'}")

    result["all_paired"] = all_ok
    summary_path = REPO_ROOT / "reports" / "multiscene_build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nbuild summary -> {summary_path}")
    if not all_ok:
        raise SystemExit("FAIL: merged dataset has unpaired image/label files.")
    print("MERGED DATASET BUILT OK.")


if __name__ == "__main__":
    main()
