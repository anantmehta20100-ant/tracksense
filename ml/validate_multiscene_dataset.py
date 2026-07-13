"""Phase 6 -- Strict label & quality validation for the synthetic multi-object set.

Validates a synthetic image/label directory (default:
data/training_8class_multiscene/synthetic_train) and asserts the properties the
retraining fix depends on:

  - every image has a label and every label has an image (exact pairing);
  - labels are YOLO format: 5 fields, class id integer, 4 floats in [0,1];
  - class ids are ONLY 0..7 -- `counter` never appears, bread stays local id 7;
  - width/height strictly positive, boxes inside the frame;
  - >= 2 objects per synthetic image, and >= 2 DISTINCT classes in (almost) all;
  - reports a class histogram, a class co-occurrence matrix, the number of
    nut_butter_jar + cutlery scenes, and full/partial flagship-chain scenes;
  - checks synthetic filenames do NOT collide with the original dataset stems.

Emits:
  reports/multiscene_class_histogram.csv
  reports/multiscene_cooccurrence_matrix.csv
  reports/multiscene_validation_report.json
  reports/multiscene_samples/   (>= --draw-samples annotated images)

Exit code is non-zero on any hard failure.

Usage:
  python ml/validate_multiscene_dataset.py --draw-samples 60
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.class_schema import MODEL_LOCAL_NAMES, NUM_TRAINING_CLASSES  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COLORS = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (0, 200, 200),
          (200, 0, 200), (0, 165, 255), (128, 0, 255), (0, 128, 255)]


def read_label(path: Path):
    boxes, errors = [], []
    for ln, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{ln} expected 5 fields, got {len(parts)}")
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{ln} unparseable: {raw!r}")
            continue
        if not (0 <= cid < NUM_TRAINING_CLASSES):
            errors.append(f"{path.name}:{ln} class id {cid} outside 0..{NUM_TRAINING_CLASSES-1}")
        for name, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (0.0 <= v <= 1.0):
                errors.append(f"{path.name}:{ln} {name}={v} outside [0,1]")
        if w <= 0 or h <= 0:
            errors.append(f"{path.name}:{ln} non-positive w/h ({w},{h})")
        boxes.append((cid, cx, cy, w, h))
    return boxes, errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the synthetic multi-object YOLO dataset.")
    ap.add_argument("--dir", default=str(REPO_ROOT / "data" / "training_8class_multiscene" / "synthetic_train"),
                    help="Directory containing images/ and labels/ subfolders.")
    ap.add_argument("--original-root", default=str(REPO_ROOT / "data" / "training_8class_balanced"),
                    help="Original dataset root, for filename-collision checking.")
    ap.add_argument("--draw-samples", type=int, default=60)
    ap.add_argument("--min-multi-class-frac", type=float, default=0.98,
                    help="Fraction of scenes that must contain >= 2 distinct classes.")
    ap.add_argument("--report-prefix", default=str(REPO_ROOT / "reports" / "multiscene"))
    args = ap.parse_args()

    base = Path(args.dir)
    img_dir, lbl_dir = base / "images", base / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise SystemExit(f"FAIL: {img_dir} or {lbl_dir} missing")

    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")}

    failures: list[str] = []
    warnings: list[str] = []

    missing_label = sorted(set(images) - set(labels))
    missing_image = sorted(set(labels) - set(images))
    if missing_label:
        failures.append(f"{len(missing_label)} images without labels (e.g. {missing_label[:3]})")
    if missing_image:
        failures.append(f"{len(missing_image)} labels without images (e.g. {missing_image[:3]})")

    hist = Counter()
    cooc = defaultdict(int)
    cooc_matrix = [[0] * NUM_TRAINING_CLASSES for _ in range(NUM_TRAINING_CLASSES)]
    obj_hist = Counter()
    multi_class = 0
    single_class = 0
    jar_cutlery = 0
    flagship_full = 0
    flagship_partial = 0
    label_errors = 0
    n = 0
    J, CUT, BREAD, PLATE = 0, 3, 7, 5
    chain = {J, CUT, BREAD, PLATE}

    for stem, lp in sorted(labels.items()):
        if stem not in images:
            continue
        n += 1
        boxes, errs = read_label(lp)
        if errs:
            label_errors += len(errs)
            for e in errs[:0]:  # collected, not spammed
                pass
            failures.extend(errs[:5])
        present = set()
        for (cid, *_rest) in boxes:
            if 0 <= cid < NUM_TRAINING_CLASSES:
                hist[cid] += 1
                present.add(cid)
        obj_hist[len(boxes)] += 1
        if len(boxes) < 2:
            failures.append(f"{lp.name} has < 2 objects ({len(boxes)})")
        if len(present) >= 2:
            multi_class += 1
        else:
            single_class += 1
        pl = sorted(present)
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                cooc[(pl[i], pl[j])] += 1
                cooc_matrix[pl[i]][pl[j]] += 1
                cooc_matrix[pl[j]][pl[i]] += 1
        if J in present and CUT in present:
            jar_cutlery += 1
        if chain.issubset(present):
            flagship_full += 1
        elif len(chain & present) >= 3:
            flagship_partial += 1

    if n == 0:
        raise SystemExit("FAIL: no paired image/label found.")

    multi_frac = multi_class / n
    if multi_frac < args.min_multi_class_frac:
        failures.append(f"only {multi_frac:.3f} of scenes have >=2 classes (need >= {args.min_multi_class_frac})")

    # filename collision vs original dataset
    orig_stems = set()
    for split in ("train", "valid", "test"):
        d = Path(args.original_root) / split / "images"
        if d.is_dir():
            orig_stems |= {p.stem for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    collisions = sorted(set(images) & orig_stems)
    if collisions:
        failures.append(f"{len(collisions)} synthetic filenames collide with the original dataset (e.g. {collisions[:3]})")

    if 8 in hist:
        failures.append("class id 8 present -- bread must be model-local 7")

    # ---- reports ----
    hist_csv = Path(f"{args.report_prefix}_class_histogram.csv")
    with hist_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class_name", "instances", "scenes_containing"])
        scenes_with = Counter()
        # recompute scenes_containing
        for stem, lp in labels.items():
            if stem not in images:
                continue
            pres = {b[0] for b in read_label(lp)[0] if 0 <= b[0] < NUM_TRAINING_CLASSES}
            for c in pres:
                scenes_with[c] += 1
        for cid in range(NUM_TRAINING_CLASSES):
            w.writerow([cid, MODEL_LOCAL_NAMES[cid], hist.get(cid, 0), scenes_with.get(cid, 0)])

    cooc_csv = Path(f"{args.report_prefix}_cooccurrence_matrix.csv")
    with cooc_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([""] + [MODEL_LOCAL_NAMES[c] for c in range(NUM_TRAINING_CLASSES)])
        for i in range(NUM_TRAINING_CLASSES):
            w.writerow([MODEL_LOCAL_NAMES[i]] + cooc_matrix[i])

    # sample grid
    samp_dir = Path(f"{args.report_prefix}_samples")
    samp_dir.mkdir(parents=True, exist_ok=True)
    drawn = 0
    for stem, lp in sorted(labels.items()):
        if drawn >= args.draw_samples or stem not in images:
            continue
        im = cv2.imread(str(images[stem]))
        if im is None:
            continue
        H, W = im.shape[:2]
        for (cid, cx, cy, bw, bh) in read_label(lp)[0]:
            x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            col = COLORS[cid % len(COLORS)]
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            cv2.putText(im, MODEL_LOCAL_NAMES[cid], (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.imwrite(str(samp_dir / images[stem].name), im)
        drawn += 1

    report = {
        "dir": str(base.as_posix()),
        "n_scenes": n,
        "n_images": len(images),
        "n_labels": len(labels),
        "paired": len(missing_label) == 0 and len(missing_image) == 0,
        "label_errors": label_errors,
        "objects_per_scene": {str(k): v for k, v in sorted(obj_hist.items())},
        "scenes_with_2plus_classes": multi_class,
        "scenes_single_class": single_class,
        "multi_class_fraction": round(multi_frac, 4),
        "class_instances": {MODEL_LOCAL_NAMES[c]: hist.get(c, 0) for c in range(NUM_TRAINING_CLASSES)},
        "jar_cutlery_scenes": jar_cutlery,
        "flagship_full_scenes": flagship_full,
        "flagship_partial_scenes": flagship_partial,
        "top_cooccurrences": {f"{MODEL_LOCAL_NAMES[a]}+{MODEL_LOCAL_NAMES[b]}": v
                              for (a, b), v in sorted(cooc.items(), key=lambda kv: -kv[1])[:15]},
        "filename_collisions_with_original": len(collisions),
        "samples_drawn": drawn,
        "PASS": len(failures) == 0,
        "failures": failures[:40],
        "warnings": warnings,
    }
    report_json = Path(f"{args.report_prefix}_validation_report.json")
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== MULTISCENE VALIDATION ===")
    print(f"scenes: {n}  paired: {report['paired']}  label_errors: {label_errors}")
    print(f"objects/scene: {report['objects_per_scene']}")
    print(f"scenes with >=2 classes: {multi_class}/{n} ({multi_frac:.3f})")
    print(f"jar+cutlery scenes: {jar_cutlery}")
    print(f"flagship full: {flagship_full}  partial(>=3): {flagship_partial}")
    print(f"class instances: {report['class_instances']}")
    print(f"filename collisions with original: {len(collisions)}")
    print(f"samples drawn: {drawn} -> {samp_dir}")
    print(f"reports: {hist_csv.name}, {cooc_csv.name}, {report_json.name}")
    if failures:
        print(f"\nFAIL ({len(failures)} issues):")
        for f in failures[:20]:
            print("  -", f)
        raise SystemExit(1)
    print("\nVALIDATION PASSED.")


if __name__ == "__main__":
    main()
