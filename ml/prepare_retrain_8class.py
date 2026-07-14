"""Convert the five retrain-v2 COCO source zips into a single isolated 8-class
YOLO dataset under data/retrain_8class/.

ISOLATED + REVERSIBLE: writes only into data/retrain_8class/. Does NOT touch any
existing data/ prep, any existing data.yaml, or the working checkpoint.

Frozen target schema (do NOT change):
    0 nut_butter_jar  1 whole_nuts  2 hand  3 cutlery
    4 chopping_board  5 plate       6 bowl  7 bread

Approved remap decisions:
  - all nut subtypes + all peanuts        -> 1 whole_nuts
  - fork/knife/spoon (metal AND plastic)  -> 3 cutlery
  - chopping-board                        -> 4 chopping_board
  - plate / mini plate / serving-plate    -> 5 plate
  - bowl / food-bowl                      -> 6 bowl
  - bread variants                        -> 7 bread
  - food-jar                              -> DROPPED (generic jar != nut_butter_jar)
  - countertop* and all irrelevant classes-> DROPPED (no target; no `counter` class)
  - peanuts CAPPED (per-split instance cap) so it does not swamp whole_nuts.

Roboflow COCO note: category id 0 in each export is an empty supercategory and
carries no annotations, so it drops naturally (and is not in any remap below).

Nothing is pseudo-labelled or fabricated; every box comes from a source label.
Images whose annotations ALL drop are skipped (no empty-label images emitted).
Run under base Python (no torch/cv2 needed; sizes come from COCO width/height).
"""
from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
DOWNLOADS = Path(r"C:/Users/tanvi/Downloads")
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "retrain_8class"

CLASS_NAMES = {
    0: "nut_butter_jar", 1: "whole_nuts", 2: "hand", 3: "cutlery",
    4: "chopping_board", 5: "plate", 6: "bowl", 7: "bread",
}

# Deterministic peanut cap: max kept instances per split (image-level selection,
# sorted by image id, accumulate until the cap is reached).
PEANUT_INSTANCE_CAP = {"train": 1500, "valid": 400, "test": 200}

# Per-dataset source-category-id -> target-id remap. Any src id NOT present here
# is DROPPED. (id 0 supercategories are intentionally absent = dropped.)
DATASETS = [
    {
        "name": "nutcls",
        "zip": "Nut classification.v1i.coco.zip",
        "remap": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
        "cap": None,
    },
    {
        "name": "peanuts",
        "zip": "peanuts.v2-release.coco.zip",
        "remap": {1: 1, 2: 1},          # with mold / without mold -> whole_nuts
        "cap": PEANUT_INSTANCE_CAP,
    },
    {
        "name": "kutensils",
        "zip": "Kitchen Utensils Recognition.v1i.coco.zip",
        "remap": {
            2: 4,                        # chopping-board
            10: 6,                       # food-bowl
            17: 3, 20: 3, 21: 3, 25: 3, 26: 3, 34: 3,  # knife/forks/spoons (metal+plastic)
            22: 5, 27: 5, 31: 5,         # mini plate / plate / serving-plate
            # 11 food-jar -> intentionally DROPPED
        },
        "cap": None,
    },
    {
        "name": "kobjdet",
        "zip": "kitchen-object-detection.v1-kitchen-final-dataset.coco.zip",
        "remap": {
            3: 6,                        # bowl
            7: 3, 8: 3, 13: 3,           # fork / knife / spoon
            10: 5,                       # plate
            # 4 countertopstone, 5 countertopwood -> DROPPED (no counter class)
        },
        "cap": None,
    },
    {
        "name": "bread",
        "zip": "Bread Detection.v4i.coco.zip",
        "remap": {1: 7, 2: 7},          # Original Classic / Jumbo -> bread
        "cap": None,
    },
]

SPLITS = ("train", "valid", "test")


def yolo_line(cat: int, x: float, y: float, w: float, h: float, iw: int, ih: int):
    """COCO absolute [x,y,w,h] (top-left) -> YOLO normalized cx cy w h. None if degenerate."""
    if w <= 0 or h <= 0 or iw <= 0 or ih <= 0:
        return None
    cx = (x + w / 2.0) / iw
    cy = (y + h / 2.0) / ih
    nw = w / iw
    nh = h / ih
    # clip to valid range
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    if nw <= 0 or nh <= 0:
        return None
    return f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def main():
    if OUT_ROOT.exists():
        print(f"[reset] removing existing {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    for split in SPLITS:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

    # stats
    img_count = defaultdict(int)                     # split -> images kept
    inst_count = defaultdict(Counter)                # split -> Counter(target_id)
    per_ds = defaultdict(lambda: defaultdict(int))   # dataset -> split -> images kept

    for ds in DATASETS:
        zpath = DOWNLOADS / ds["zip"]
        if not zpath.exists():
            raise FileNotFoundError(f"missing source zip: {zpath}")
        with zipfile.ZipFile(zpath) as zf:
            names = set(zf.namelist())
            for split in SPLITS:
                jname = f"{split}/_annotations.coco.json"
                if jname not in names:
                    continue  # e.g. kobjdet has no valid split
                data = json.loads(zf.read(jname))
                images = {im["id"]: im for im in data["images"]}
                anns_by_img = defaultdict(list)
                for a in data["annotations"]:
                    anns_by_img[a["image_id"]].append(a)

                cap = ds["cap"][split] if ds["cap"] else None
                kept_for_cap = 0

                for img_id in sorted(images):                 # deterministic
                    im = images[img_id]
                    iw, ih = int(im["width"]), int(im["height"])
                    lines = []
                    for a in anns_by_img.get(img_id, []):
                        tgt = ds["remap"].get(a["category_id"])
                        if tgt is None:
                            continue
                        x, y, w, h = a["bbox"]
                        ln = yolo_line(tgt, x, y, w, h, iw, ih)
                        if ln is not None:
                            lines.append((tgt, ln))
                    if not lines:
                        continue  # all-dropped -> skip (no empty-label images)

                    # peanut cap: stop taking new images once instance cap reached
                    if cap is not None and kept_for_cap >= cap:
                        break
                    if cap is not None:
                        kept_for_cap += len(lines)

                    src_file = f"{split}/{im['file_name']}"
                    stem = f"{ds['name']}__{Path(im['file_name']).stem}"
                    ext = Path(im["file_name"]).suffix or ".jpg"
                    out_img = OUT_ROOT / split / "images" / f"{stem}{ext}"
                    out_lbl = OUT_ROOT / split / "labels" / f"{stem}.txt"

                    with open(out_img, "wb") as fh:
                        fh.write(zf.read(src_file))
                    out_lbl.write_text("\n".join(ln for _, ln in lines) + "\n")

                    img_count[split] += 1
                    per_ds[ds["name"]][split] += 1
                    for tgt, _ in lines:
                        inst_count[split][tgt] += 1

    # data.yaml (isolated -- retrain set only)
    yaml_path = OUT_ROOT / "data.yaml"
    with open(yaml_path, "w") as fh:
        fh.write("# TrackSense retrain-v2 ISOLATED 8-class YOLO set (new weak-class data).\n")
        fh.write("# Built by ml/prepare_retrain_8class.py from 5 COCO source zips.\n")
        fh.write("# NOT the training set on its own -- must be MERGED with the existing\n")
        fh.write("# multiscene set before training (catastrophic-forgetting guard).\n")
        fh.write("# Frozen schema: bread=7, no counter.\n\n")
        fh.write("path: ../data/retrain_8class\n")
        fh.write("train: train/images\n")
        fh.write("val: valid/images\n")
        fh.write("test: test/images\n\n")
        fh.write("names:\n")
        for i in range(8):
            fh.write(f"  {i}: {CLASS_NAMES[i]}\n")

    # ---- report ----
    print("\n=== retrain_8class conversion complete ===")
    print(f"output: {OUT_ROOT}")
    print("\nimages kept per dataset/split:")
    for name in (d["name"] for d in DATASETS):
        row = per_ds[name]
        print(f"  {name:10s} train={row['train']:4d} valid={row['valid']:4d} test={row['test']:4d}")
    print("\nimages per split:", {s: img_count[s] for s in SPLITS},
          "TOTAL =", sum(img_count.values()))
    print("\ninstances per class (all splits):")
    total = Counter()
    for s in SPLITS:
        total.update(inst_count[s])
    for i in range(8):
        parts = " ".join(f"{s}={inst_count[s].get(i,0)}" for s in SPLITS)
        print(f"  {i} {CLASS_NAMES[i]:15s} total={total.get(i,0):6d}   ({parts})")
    print(f"\ndata.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
