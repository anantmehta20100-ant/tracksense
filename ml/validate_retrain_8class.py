"""Integrity check for a YOLO 8-class dataset dir (isolated retrain set or the
eventual merged set). Reports and exits non-zero on failure.

Checks: 1:1 image/label pairing, no orphan labels, no empty-label files,
all class ids in 0..7, no `counter`, bread=7 present as id 7 only.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

VALID_IDS = set(range(8))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_split(split_dir: Path):
    problems = []
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not img_dir.exists():
        return problems, Counter(), 0  # split absent is allowed
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    lbls = {p.stem: p for p in lbl_dir.iterdir() if p.suffix == ".txt"} if lbl_dir.exists() else {}

    for stem in imgs.keys() - lbls.keys():
        problems.append(f"[{split_dir.name}] image without label: {stem}")
    for stem in lbls.keys() - imgs.keys():
        problems.append(f"[{split_dir.name}] orphan label without image: {stem}")

    cls = Counter()
    for stem, lp in lbls.items():
        rows = [r for r in lp.read_text().splitlines() if r.strip()]
        if not rows:
            problems.append(f"[{split_dir.name}] empty-label file: {stem}")
            continue
        for r in rows:
            parts = r.split()
            if len(parts) != 5:
                problems.append(f"[{split_dir.name}] bad row ({len(parts)} fields) in {stem}: {r!r}")
                continue
            try:
                cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                problems.append(f"[{split_dir.name}] non-numeric row in {stem}: {r!r}")
                continue
            if cid not in VALID_IDS:
                problems.append(f"[{split_dir.name}] class id {cid} out of 0..7 in {stem}")
            if any(not (0.0 <= c <= 1.0) for c in coords):
                problems.append(f"[{split_dir.name}] coord out of [0,1] in {stem}: {r!r}")
            cls[cid] += 1
    return problems, cls, len(imgs)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "data" / "retrain_8class")
    print(f"validating: {root}")
    all_problems = []
    total_cls = Counter()
    total_imgs = 0
    for split in ("train", "valid", "test"):
        probs, cls, n = check_split(root / split)
        all_problems += probs
        total_cls.update(cls)
        total_imgs += n
        print(f"  {split:6s} images={n:4d} instances={sum(cls.values()):5d} "
              f"classes={dict(sorted(cls.items()))}")

    print(f"\ntotal images={total_imgs} instances={sum(total_cls.values())}")
    print("class totals:", {k: total_cls[k] for k in sorted(total_cls)})
    max_id = max(total_cls) if total_cls else -1
    print(f"max class id present = {max_id} (must be <=7); 'counter' impossible (no id 8)")

    if all_problems:
        print(f"\nFAILED: {len(all_problems)} problem(s):")
        for p in all_problems[:50]:
            print("  -", p)
        if len(all_problems) > 50:
            print(f"  ... and {len(all_problems) - 50} more")
        sys.exit(1)
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
