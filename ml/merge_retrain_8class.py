"""Step 4 -- merge the isolated retrain-v2 set with the EXISTING 8-class
multiscene set into a new combined YOLO dataset (catastrophic-forgetting guard).

Path-agnostic so it runs on Colab or locally once the existing set is re-sourced.
Writes ONLY into --out (default data/merged_retrain_v2/). Never modifies the
existing multiscene set, the isolated retrain set, or any checkpoint.

Both inputs must be YOLO layout: <root>/{train,valid,test}/{images,labels}.
Files are copied (not moved) with source-namespaced prefixes so there are no
filename collisions. Reports per-split / per-class image + instance counts and
flags under/over-represented classes.

Example:
  python ml/merge_retrain_8class.py \
    --existing /content/training_8class_multiscene \
    --retrain  /content/data/retrain_8class \
    --out      /content/data/merged_retrain_v2
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = {
    0: "nut_butter_jar", 1: "whole_nuts", 2: "hand", 3: "cutlery",
    4: "chopping_board", 5: "plate", 6: "bowl", 7: "bread",
}
SPLITS = ("train", "valid", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_split(src_root: Path, split: str, tag: str, out_root: Path,
               imgs_kept: dict, inst: dict):
    """Copy one split from one source into the merged out, prefixing with `tag`."""
    img_dir = src_root / split / "images"
    lbl_dir = src_root / split / "labels"
    if not img_dir.exists():
        return
    out_img = out_root / split / "images"
    out_lbl = out_root / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() not in IMG_EXTS:
            continue
        lp = lbl_dir / (ip.stem + ".txt")
        if not lp.exists():
            # keep merged set clean: skip images without a label
            continue
        rows = [r for r in lp.read_text().splitlines() if r.strip()]
        if not rows:
            continue  # skip empty-label images
        stem = f"{tag}__{ip.stem}"
        shutil.copy2(ip, out_img / f"{stem}{ip.suffix.lower()}")
        (out_lbl / f"{stem}.txt").write_text("\n".join(rows) + "\n")
        imgs_kept[split] += 1
        for r in rows:
            try:
                inst[split][int(r.split()[0])] += 1
            except (ValueError, IndexError):
                pass


def write_yaml(out_root: Path):
    yp = out_root / "data.yaml"
    with open(yp, "w") as fh:
        fh.write("# TrackSense retrain-v2 MERGED 8-class set (existing multiscene + new weak-class data).\n")
        fh.write("# Built by ml/merge_retrain_8class.py. Frozen schema: bread=7, no counter.\n\n")
        fh.write("path: .\n")
        fh.write("train: train/images\n")
        fh.write("val: valid/images\n")
        fh.write("test: test/images\n\n")
        fh.write("names:\n")
        for i in range(8):
            fh.write(f"  {i}: {CLASS_NAMES[i]}\n")
    return yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", required=True,
                    help="root of existing training_8class_multiscene YOLO set")
    ap.add_argument("--retrain", required=True,
                    help="root of isolated data/retrain_8class YOLO set")
    ap.add_argument("--out", required=True, help="output merged dataset root")
    args = ap.parse_args()

    existing = Path(args.existing)
    retrain = Path(args.retrain)
    out_root = Path(args.out)

    if not existing.exists():
        raise SystemExit(
            f"ERROR: existing set not found: {existing}\n"
            "The merge is the catastrophic-forgetting guard -- re-source the "
            "original training_8class_multiscene set before running this.")
    if not retrain.exists():
        raise SystemExit(f"ERROR: retrain set not found: {retrain}")

    if out_root.exists():
        print(f"[reset] removing existing {out_root}")
        shutil.rmtree(out_root)

    imgs_kept = defaultdict(int)
    inst = defaultdict(Counter)

    for split in SPLITS:
        copy_split(existing, split, "existing", out_root, imgs_kept, inst)
        copy_split(retrain, split, "retrain", out_root, imgs_kept, inst)

    yp = write_yaml(out_root)

    # ---- report ----
    print("\n=== merged_retrain_v2 ===")
    print(f"output: {out_root}")
    total_inst = Counter()
    total_imgs = 0
    for split in SPLITS:
        total_inst.update(inst[split])
        total_imgs += imgs_kept[split]
        print(f"  {split:6s} images={imgs_kept[split]:5d} instances={sum(inst[split].values()):6d}")
    grand = sum(total_inst.values())
    print(f"\nTOTAL images={total_imgs}  instances={grand}")
    print("\nper-class instances (all splits):")
    for i in range(8):
        n = total_inst.get(i, 0)
        pct = (100.0 * n / grand) if grand else 0.0
        print(f"  {i} {CLASS_NAMES[i]:15s} {n:7d}  ({pct:5.1f}%)")

    # imbalance flags
    print("\nbalance flags:")
    flagged = False
    for i in range(8):
        n = total_inst.get(i, 0)
        pct = (100.0 * n / grand) if grand else 0.0
        if n == 0:
            print(f"  !! class {i} {CLASS_NAMES[i]}: ZERO instances -- MISSING from merged set")
            flagged = True
        elif pct < 3.0:
            print(f"  !  class {i} {CLASS_NAMES[i]}: under-represented ({pct:.1f}%, {n})")
            flagged = True
        elif pct > 40.0:
            print(f"  !  class {i} {CLASS_NAMES[i]}: over-represented ({pct:.1f}%, {n})")
            flagged = True
    if not flagged:
        print("  (none)")
    print(f"\nmerged data.yaml -> {yp}")


if __name__ == "__main__":
    main()
