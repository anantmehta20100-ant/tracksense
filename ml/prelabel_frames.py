"""Draft YOLO labels for captured frames, so labelling is correcting, not drawing.

    python ml/prelabel_frames.py --images data/own_frames/session1

Writes next to the images folder:
  <dir>/labels/<frame>.txt   one YOLO box per detection (model-local class ids)
  <dir>/classes.txt          class names in id order (labelImg / X-AnyLabeling)
  <dir>/data.yaml            names + val path (Roboflow upload, ml eval scripts)
  <dir>/drafts.json          fingerprint of each draft, so ml/build_owncam_dataset.py
                             can warn about label files nobody has edited

Every draft box must be checked by a person before the frames are used as a
test set: a model graded on its own unchecked guesses scores itself perfect.
Existing label files are never overwritten unless --overwrite, so hand
corrections survive a re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import YOLO_IMGSZ, YOLO_MODEL_PATH, resolve_device  # noqa: E402

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="folder of frames (moved into <folder>/images/)")
    parser.add_argument("--weights", default=YOLO_MODEL_PATH)
    parser.add_argument("--conf", type=float, default=0.30,
                        help="draft threshold: lower = fewer missed objects to draw, more wrong boxes to delete")
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.images).resolve()
    images_dir = root / "images" if (root / "images").is_dir() else root
    if images_dir == root:  # flat folder of frames -> standard images/ + labels/ layout
        images_dir = root / "images"
        images_dir.mkdir(exist_ok=True)
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.move(str(p), images_dir / p.name)
    labels_dir = root / "labels"
    labels_dir.mkdir(exist_ok=True)
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"no images in {images_dir}")

    from ultralytics import YOLO

    device = resolve_device(args.device)
    model = YOLO(args.weights)
    names = {int(k): v for k, v in model.names.items()}
    print(f"{len(images)} images, weights {args.weights}, device {device}, conf {args.conf}")
    print(f"classes: {names}")

    drafts_file = root / "drafts.json"
    drafts = json.loads(drafts_file.read_text()) if drafts_file.is_file() else {}
    written = skipped = boxes = 0
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if label.exists() and not args.overwrite:
            skipped += 1
            continue
        result = model(str(image), imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for c, (x, y, w, h) in zip(result.boxes.cls.tolist(), result.boxes.xywhn.tolist())]
        label.write_text("\n".join(lines) + ("\n" if lines else ""))
        drafts[image.stem] = hashlib.sha1(label.read_bytes()).hexdigest()
        written, boxes = written + 1, boxes + len(lines)

    drafts_file.write_text(json.dumps(drafts, indent=0))
    (root / "classes.txt").write_text("\n".join(names[i] for i in sorted(names)) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"path": root.as_posix(), "train": "images", "val": "images", "names": names}, sort_keys=False))
    print(f"drafted {written} label files ({boxes} boxes), kept {skipped} existing")
    print(f"next: correct every box in {root} (e.g. X-AnyLabeling / labelImg with classes.txt, "
          "or upload images+labels+data.yaml to Roboflow), then run evaluate/evaluate_detector.py")


if __name__ == "__main__":
    main()
