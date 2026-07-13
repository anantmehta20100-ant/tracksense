"""Phase 3 -- Build an object-instance crop bank from the existing 8-class TRAIN set.

Every image in data/training_8class_balanced is single-object-type (audited: 0
images contain more than one distinct class). To teach YOLO multi-object
co-occurrence we first need a bank of individual object cut-outs that a scene
generator can composite together (ml/generate_multiscene_yolo_data.py).

For every labelled box in the TRAIN split we:
  - read the source image, convert the normalized YOLO bbox to pixels,
  - add a small padding margin (clamped to the image),
  - crop the padded region and downscale to a manageable max dimension,
  - bake a *feathered alpha* channel so the rectangular cut-out blends into a
    new background instead of showing a hard seam (rectangular crops only -- no
    segmentation masks are required, per the task),
  - record where the true object sits inside the padded crop (inner-object
    fractions) so the generator can emit a TIGHT YOLO label, not a padded one,
  - save each crop as an RGBA PNG under data/object_crop_bank/<id>_<name>/.

Quality filters skip tiny / invalid / near-zero-area / mostly-out-of-frame
boxes, and flag (but keep, unless --drop-near-full) near-full-frame crops that
would carry too much of their own background.

Schema is READ-ONLY here: class ids are the 8 model-local ids from
ml/class_schema.py. No labels are rewritten, nothing in the source dataset is
modified. Reproducible with --seed (default 42).

Usage:
  python ml/create_object_crop_bank.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.class_schema import MODEL_LOCAL_NAMES, NUM_TRAINING_CLASSES  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def parse_label(label_path: Path):
    """Yield (class_id, cx, cy, w, h) tuples from a YOLO label file."""
    rows = []
    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        rows.append((cid, cx, cy, w, h))
    return rows


def feather_alpha(w: int, h: int, band_frac: float) -> np.ndarray:
    """Alpha ramp: full opacity in the interior, linearly fading toward edges."""
    band = max(1, int(round(min(w, h) * band_frac)))
    xr = np.minimum(np.arange(w), np.arange(w)[::-1])
    yr = np.minimum(np.arange(h), np.arange(h)[::-1])
    ax = np.clip(xr / band, 0.0, 1.0)
    ay = np.clip(yr / band, 0.0, 1.0)
    a = np.minimum.outer(ay, ax)
    # keep a solid core; only the outer band fades
    return (a * 255.0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an object-instance crop bank for multi-scene synthesis.")
    ap.add_argument("--src-root", default=str(REPO_ROOT / "data" / "training_8class_balanced"))
    ap.add_argument("--split", default="train", help="Source split to crop from (train only, to avoid val/test leakage).")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "object_crop_bank"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class-cap", type=int, default=600, help="Max crops kept per class (minority classes keep all).")
    ap.add_argument("--pad-frac", type=float, default=0.06, help="Padding added around each bbox, as a fraction of bbox size.")
    ap.add_argument("--feather-frac", type=float, default=0.10, help="Edge-feather band width as a fraction of crop min-dim.")
    ap.add_argument("--max-dim", type=int, default=320, help="Downscale crops so the longer side is at most this many px.")
    ap.add_argument("--min-crop-px", type=int, default=24, help="Skip crops whose padded region is smaller than this on either side.")
    ap.add_argument("--min-area-frac", type=float, default=0.0008, help="Skip boxes whose area is below this fraction of the source image.")
    ap.add_argument("--near-full-frac", type=float, default=0.85, help="Flag boxes whose area exceeds this fraction of the image as near-full-frame.")
    ap.add_argument("--drop-near-full", action="store_true", help="Actually discard near-full-frame crops instead of just flagging them.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src_root = Path(args.src_root)
    images_dir = src_root / args.split / "images"
    labels_dir = src_root / args.split / "labels"
    out_root = Path(args.out)

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"FAIL: missing {images_dir} or {labels_dir}")

    for cid in range(NUM_TRAINING_CLASSES):
        (out_root / f"{cid}_{MODEL_LOCAL_NAMES[cid]}").mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    rng.shuffle(label_files)
    print(f"Source: {images_dir}  ({len(label_files)} label files)")
    print(f"Output: {out_root}  seed={args.seed}  per_class_cap={args.per_class_cap}  max_dim={args.max_dim}")

    kept = Counter()
    stats = Counter()
    near_full_by_class = Counter()
    rows: list[dict] = []

    for lf in label_files:
        boxes = parse_label(lf)
        if not boxes:
            continue
        # Single-object dataset: every box in a file shares one class. Still,
        # respect the cap by class of the file's first box.
        file_class = boxes[0][0]
        if file_class < 0 or file_class >= NUM_TRAINING_CLASSES:
            stats["bad_class_id"] += 1
            continue
        if kept[file_class] >= args.per_class_cap and all(b[0] == file_class for b in boxes):
            continue

        img_path = find_image(images_dir, lf.stem)
        if img_path is None:
            stats["missing_image"] += 1
            continue
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            stats["unreadable_image"] += 1
            continue
        W, H = im.size

        for idx, (cid, cx, cy, bw, bh) in enumerate(boxes):
            if cid < 0 or cid >= NUM_TRAINING_CLASSES:
                stats["bad_class_id"] += 1
                continue
            if kept[cid] >= args.per_class_cap:
                continue
            if bw <= 0 or bh <= 0:
                stats["nonpositive_box"] += 1
                continue
            area_frac = bw * bh
            if area_frac < args.min_area_frac:
                stats["too_small_area"] += 1
                continue

            # normalized -> pixel bbox
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            # reject boxes whose center is outside the frame / mostly out
            if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
                stats["out_of_frame"] += 1
                continue

            near_full = area_frac >= args.near_full_frac
            if near_full:
                near_full_by_class[cid] += 1
                if args.drop_near_full:
                    stats["near_full_dropped"] += 1
                    continue

            pad_x = bw * W * args.pad_frac
            pad_y = bh * H * args.pad_frac
            px1 = int(round(max(0, x1 - pad_x)))
            py1 = int(round(max(0, y1 - pad_y)))
            px2 = int(round(min(W, x2 + pad_x)))
            py2 = int(round(min(H, y2 + pad_y)))
            cw, ch = px2 - px1, py2 - py1
            if cw < args.min_crop_px or ch < args.min_crop_px:
                stats["too_small_px"] += 1
                continue

            # inner-object fractions inside the padded crop (for a TIGHT label)
            ox1 = max(0.0, (x1 - px1) / cw)
            oy1 = max(0.0, (y1 - py1) / ch)
            ox2 = min(1.0, (x2 - px1) / cw)
            oy2 = min(1.0, (y2 - py1) / ch)
            if ox2 - ox1 <= 0 or oy2 - oy1 <= 0:
                stats["degenerate_inner"] += 1
                continue

            crop = im.crop((px1, py1, px2, py2)).convert("RGB")
            # downscale so the longer side <= max_dim
            scale = min(1.0, args.max_dim / max(crop.width, crop.height))
            if scale < 1.0:
                crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)

            arr = np.array(crop)
            alpha = feather_alpha(crop.width, crop.height, args.feather_frac)
            rgba = np.dstack([arr, alpha])
            out_im = Image.fromarray(rgba, mode="RGBA")

            cls_dir = out_root / f"{cid}_{MODEL_LOCAL_NAMES[cid]}"
            crop_name = f"{cid}_{MODEL_LOCAL_NAMES[cid]}__{lf.stem[:60]}__b{idx}.png"
            crop_path = cls_dir / crop_name
            out_im.save(crop_path, optimize=True)

            kept[cid] += 1
            stats["kept"] += 1
            rows.append({
                "crop_path": str(crop_path.relative_to(REPO_ROOT).as_posix()),
                "class_id": cid,
                "class_name": MODEL_LOCAL_NAMES[cid],
                "source_split": args.split,
                "source_image": img_path.name,
                "source_label": lf.name,
                "orig_cx": round(cx, 6), "orig_cy": round(cy, 6),
                "orig_w": round(bw, 6), "orig_h": round(bh, 6),
                "orig_area_frac": round(area_frac, 6),
                "crop_w": out_im.width, "crop_h": out_im.height,
                "obj_l": round(ox1, 6), "obj_t": round(oy1, 6),
                "obj_r": round(ox2, 6), "obj_b": round(oy2, 6),
                "near_full_frame": int(near_full),
                "n_boxes_in_source": len(boxes),
            })

    # write index + summary
    index_csv = out_root / "crop_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["crop_path"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "seed": args.seed,
        "source_root": str(src_root.as_posix()),
        "source_split": args.split,
        "per_class_cap": args.per_class_cap,
        "max_dim": args.max_dim,
        "pad_frac": args.pad_frac,
        "feather_frac": args.feather_frac,
        "total_crops": int(stats["kept"]),
        "crops_per_class": {f"{cid}_{MODEL_LOCAL_NAMES[cid]}": int(kept[cid]) for cid in range(NUM_TRAINING_CLASSES)},
        "near_full_frame_per_class": {f"{cid}_{MODEL_LOCAL_NAMES[cid]}": int(near_full_by_class[cid]) for cid in range(NUM_TRAINING_CLASSES)},
        "skipped": {k: int(v) for k, v in stats.items() if k != "kept"},
    }
    (out_root / "crop_bank_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== CROP BANK SUMMARY ===")
    print(f"total crops kept: {summary['total_crops']}")
    for cid in range(NUM_TRAINING_CLASSES):
        name = MODEL_LOCAL_NAMES[cid]
        print(f"  {cid} {name:16} kept={kept[cid]:5d}  near_full={near_full_by_class[cid]}")
    print("skipped:", summary["skipped"])
    print(f"index : {index_csv}")
    print(f"summary: {out_root / 'crop_bank_summary.json'}")
    if summary["total_crops"] == 0:
        raise SystemExit("FAIL: no crops produced.")


if __name__ == "__main__":
    main()
