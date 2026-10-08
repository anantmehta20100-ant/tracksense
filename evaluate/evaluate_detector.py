"""Score one or more YOLO checkpoints on a hand-labelled set from the real camera.

    python evaluate/evaluate_detector.py --data data/own_frames/session1/data.yaml \
        --weights model/checkpoints/tracksense_8class_best.pt other.pt

Prints per-class P, R, mAP50, mAP50-95 side by side, plus inference ms/image.
Every checkpoint must use the same class names as the data yaml; otherwise its
class ids mean different things and the numbers are garbage, so it is refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import YOLO_IMGSZ, resolve_device  # noqa: E402


def load_names(data_yaml: Path) -> dict[int, str]:
    names = yaml.safe_load(data_yaml.read_text())["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def evaluate(weights: str, data_yaml: Path, split: str, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    model_names = {int(k): v for k, v in model.names.items()}
    data_names = load_names(data_yaml)
    if model_names != data_names:
        raise SystemExit(f"{weights}: class names {model_names} do not match {data_yaml}: {data_names}")
    metrics = model.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device,
                        plots=False, verbose=False)
    box = metrics.box
    rows = {}
    for i, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        rows[data_names[int(cls)]] = (p, r, ap50, ap)
    rows["all"] = (box.mp, box.mr, box.map50, box.map)
    return {"rows": rows, "ms": metrics.speed.get("inference", float("nan"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="data.yaml of the labelled set")
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    device = resolve_device(args.device)
    results = {w: evaluate(w, data_yaml, args.split, args.imgsz, device) for w in args.weights}

    classes = [n for n in load_names(data_yaml).values() if any(n in r["rows"] for r in results.values())]
    print(f"\n{data_yaml} [{args.split}]  imgsz={args.imgsz}  device={device}")
    for w, r in results.items():
        print(f"\n{w}   ({r['ms']:.1f} ms/image inference)")
        print(f"  {'class':<16}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
        for name in classes + ["all"]:
            if name in r["rows"]:
                p, rec, ap50, ap = r["rows"][name]
                print(f"  {name:<16}{p:>7.3f}{rec:>7.3f}{ap50:>8.3f}{ap:>10.3f}")
            else:
                print(f"  {name:<16}{'(no labels in this set)':>32}")


if __name__ == "__main__":
    main()
