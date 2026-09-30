"""Merge two Roboflow YOLO exports into one fine-tuning set for cutlery.

Sources (as written by download.py):
  ds1  datasets/11-classes-v2      only images whose ORIGINAL filename starts
                                   with IMG_1 (top-down counter photos) are kept
  ds2  datasets/cutlery-4dlwb-v4   every image, train split only

Output: datasets/combined/{images,labels}/{train,val} + data.yaml (absolute).

  - one class list: fork, knife, spoon, cup, plate (SmallFork->fork,
    SmallSpoon->spoon); every other box is dropped, so scissors etc. become
    unlabelled background
  - Roboflow renames files to <orig>_<ext>.rf.<hash>.jpg; the part before
    ".rf." is the original photo, used as the grouping key
  - ds1 is pooled across its train/valid/test (the export leaks the same photo
    into several splits), de-duplicated per original photo, then re-split
    ~80/20 train/val grouped by original photo
  - ds2 goes to train only (phone-burst near-duplicates would leak into val)

  --sheet N draws the boxes on N random output images per source into
  datasets/combined/_sheets/<source>.jpg for a manual label check.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

CLASSES = ["fork", "knife", "spoon", "cup", "plate"]
ALIASES = {"smallfork": "fork", "smallspoon": "spoon"}
SOURCE_SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
COLORS = {"fork": (0, 200, 255), "knife": (0, 0, 255), "spoon": (255, 120, 0),
          "cup": (200, 0, 200), "plate": (0, 200, 0)}


def original_name(image: Path) -> str:
    return image.name.split(".rf.")[0]


def load_names(root: Path) -> list[str]:
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def target_id(name: str) -> int | None:
    key = name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    key = ALIASES.get(key, key)
    return CLASSES.index(key) if key in CLASSES else None


def read_boxes(label: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label.is_file():
        return boxes
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, vals = int(float(parts[0])), [float(v) for v in parts[1:]]
        if len(vals) > 4:  # polygon row -> enclosing box
            xs, ys = vals[0::2], vals[1::2]
            vals = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
        boxes.append((cls, *vals[:4]))
    return boxes


def scan(root: Path) -> list[dict]:
    items = []
    for split in SOURCE_SPLITS:
        images_dir = root / split / "images"
        if not images_dir.is_dir():
            continue
        for image in sorted(images_dir.iterdir()):
            if image.suffix.lower() in IMAGE_EXTENSIONS:
                label = root / split / "labels" / f"{image.stem}.txt"
                items.append({"image": image, "split": split, "orig": original_name(image),
                              "boxes": read_boxes(label)})
    return items


def print_source_report(tag: str, root: Path, items: list[dict], names: list[str]) -> None:
    print(f"\n=== {tag}: {root}")
    print(f"classes ({len(names)}): {names}")
    split_counts = Counter(it["split"] for it in items)
    print("split images: " + ", ".join(f"{s}={split_counts.get(s, 0)}" for s in SOURCE_SPLITS)
          + f", total={len(items)}")
    print_class_table(items, names)


def print_class_table(items: list[dict], names: list[str]) -> None:
    images, boxes = Counter(), Counter()
    for it in items:
        classes = [names[b[0]] if b[0] < len(names) else f"id{b[0]}" for b in it["boxes"]]
        boxes.update(classes)
        images.update(set(classes))
    empty = sum(1 for it in items if not it["boxes"])
    print(f"  {'class':<14}{'images':>8}{'boxes':>8}")
    for name in sorted(boxes, key=lambda n: -boxes[n]):
        print(f"  {name:<14}{images[name]:>8}{boxes[name]:>8}")
    print(f"  {'(no boxes)':<14}{empty:>8}")


def leakage(items: list[dict]) -> dict[str, set[str]]:
    splits_by_orig = defaultdict(set)
    for it in items:
        splits_by_orig[it["orig"]].add(it["split"])
    return {orig: s for orig, s in splits_by_orig.items() if len(s) > 1}


def dedupe(items: list[dict]) -> tuple[list[dict], int, list[str]]:
    """One copy per original photo; flags copies whose labels disagree."""
    by_orig = defaultdict(list)
    for it in items:
        by_orig[it["orig"]].append(it)
    kept, conflicts = [], []
    for orig in sorted(by_orig):
        copies = sorted(by_orig[orig], key=lambda it: SOURCE_SPLITS.index(it["split"]))
        kept.append(copies[0])
        label_sets = {tuple(sorted((b[0], *(round(v, 3) for v in b[1:])) for b in c["boxes"])) for c in copies}
        if len(label_sets) > 1:
            conflicts.append(orig)
    return kept, len(items) - len(kept), conflicts


def remap(items: list[dict], names: list[str]) -> list[dict]:
    mapping = {i: target_id(n) for i, n in enumerate(names)}
    out = []
    for it in items:
        boxes = [(mapping[b[0]], *b[1:]) for b in it["boxes"] if mapping.get(b[0]) is not None]
        out.append({**it, "boxes": boxes})
    return out


def write_split(items: list[dict], out: Path, split: str, tag: str) -> None:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for it in items:
        name = f"{tag}__{it['image'].name}"
        shutil.copy2(it["image"], out / "images" / split / name)
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in it["boxes"]]
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def contact_sheet(out: Path, tag: str, count: int, seed: int, tile: int = 400, cols: int = 5) -> Path | None:
    images = sorted(p for p in (out / "images").glob(f"*/{tag}__*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        return None
    picks = random.Random(seed).sample(images, min(count, len(images)))
    tiles = []
    for image in picks:
        frame = cv2.imread(str(image))
        h, w = frame.shape[:2]
        label = out / "labels" / image.parent.name / f"{image.stem}.txt"
        for c, x, y, bw, bh in read_boxes(label):
            color = COLORS[CLASSES[c]]
            p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
            p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
            cv2.rectangle(frame, p1, p2, color, max(2, w // 300))
            cv2.putText(frame, CLASSES[c], (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, w / 1200), color, max(1, w // 600))
        scale = tile / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        canvas = np.full((tile + 36, tile, 3), 255, np.uint8)
        canvas[:frame.shape[0], :frame.shape[1]] = frame
        short = image.name.split("__", 1)[1].split(".rf.")[0]
        cv2.putText(canvas, f"{image.parent.name}/{short}"[:48], (4, tile + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    path = out / "_sheets" / f"{tag}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    print(f"\ncontact sheet {tag}: {path}")
    for image in picks:
        print(f"  {image.parent.name}/{image.name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ds1", default="datasets/11-classes-v2")
    parser.add_argument("--ds2", default="datasets/cutlery-4dlwb-v4")
    parser.add_argument("--out", default="datasets/combined")
    parser.add_argument("--prefix", default="IMG_1", help="keep ds1 images whose original name starts with this")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sheet", type=int, default=0, help="draw N random images per source into _sheets/")
    args = parser.parse_args()

    ds1, ds2, out = Path(args.ds1).resolve(), Path(args.ds2).resolve(), Path(args.out).resolve()
    if out in (ds1, ds2):
        raise SystemExit("--out must differ from the source datasets")

    names1, names2 = load_names(ds1), load_names(ds2)
    items1, items2 = scan(ds1), scan(ds2)
    print_source_report("ds1", ds1, items1, names1)
    leaks = leakage(items1)
    print(f"  original photos in >1 split (whole ds1): {len(leaks)}")
    for orig in sorted(leaks)[:5]:
        print(f"    {orig}: {sorted(leaks[orig])}")
    print_source_report("ds2", ds2, items2, names2)

    subset = [it for it in items1 if it["orig"].startswith(args.prefix)]
    if not subset:
        sample = ", ".join(it["image"].name for it in items1[:5])
        raise SystemExit(f"no ds1 images with original prefix {args.prefix!r}; e.g. {sample}")
    print(f"\n=== ds1 subset: original name starts with {args.prefix!r}")
    split_counts = Counter(it["split"] for it in subset)
    print("split images: " + ", ".join(f"{s}={split_counts.get(s, 0)}" for s in SOURCE_SPLITS)
          + f", total={len(subset)}")
    print_class_table(subset, names1)
    subset, dropped, conflicts = dedupe(subset)
    print(f"  duplicate copies dropped: {dropped} -> {len(subset)} unique photos")
    if conflicts:
        print(f"  copies with DIFFERENT labels ({len(conflicts)}, first kept by split order train>valid>test): "
              + ", ".join(conflicts[:10]))

    subset, items2 = remap(subset, names1), remap(items2, names2)
    print("\nclass mapping:")
    for tag, names in (("ds1", names1), ("ds2", names2)):
        print(f"  {tag}: " + ", ".join(f"{n}->{CLASSES[t] if (t := target_id(n)) is not None else 'DROP'}"
                                       for n in names))

    origs = sorted({it["orig"] for it in subset})
    random.Random(args.seed).shuffle(origs)
    val_origs = set(origs[:round(len(origs) * args.val_frac)])
    ds1_train = [it for it in subset if it["orig"] not in val_origs]
    ds1_val = [it for it in subset if it["orig"] in val_origs]

    if out.exists():
        shutil.rmtree(out)
    write_split(ds1_train, out, "train", "ds1")
    write_split(ds1_val, out, "val", "ds1")
    write_split(items2, out, "train", "ds2")

    data = {"path": out.as_posix(), "train": (out / "images" / "train").as_posix(),
            "val": (out / "images" / "val").as_posix(), "names": dict(enumerate(CLASSES))}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    print(f"\n=== combined: {out}")
    splits = {"train": ds1_train + items2, "val": ds1_val}
    print(f"  {'':<8}{'images':>8}{'ds1':>6}{'ds2':>6}{'empty':>7}" + "".join(f"{c:>8}" for c in CLASSES))
    for split, items in splits.items():
        boxes = Counter(b[0] for it in items for b in it["boxes"])
        n1 = len(ds1_train if split == "train" else ds1_val)
        empty = sum(1 for it in items if not it["boxes"])
        print(f"  {split:<8}{len(items):>8}{n1:>6}{len(items) - n1:>6}{empty:>7}"
              + "".join(f"{boxes[i]:>8}" for i in range(len(CLASSES))))
    assert not {it["orig"] for it in ds1_train} & {it["orig"] for it in ds1_val}, "photo in both splits"
    print(f"  no original photo in both splits; data.yaml -> {out / 'data.yaml'}")

    if args.sheet:
        for tag in ("ds1", "ds2"):
            contact_sheet(out, tag, args.sheet, args.seed)


if __name__ == "__main__":
    main()
