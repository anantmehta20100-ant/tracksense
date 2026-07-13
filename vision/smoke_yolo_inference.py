"""Lightweight YOLO smoke inference (optional; no dashboard/backend).

Loads the 8-class checkpoint, runs detection on ONE image or a few camera
frames, prints detected class names / confidences / boxes, and saves an
annotated frame to reports/yolo_smoke/. It first validates the checkpoint's
class schema (via vision/validate_yolo_checkpoint.check_schema) and refuses to
run on a wrong/old model.

It does NOT retrain, modify weights, or touch the dashboard.

CLI:
    python vision/smoke_yolo_inference.py --model model/checkpoints/tracksense_8class_best.pt --image path/to/img.jpg
    python vision/smoke_yolo_inference.py --model model/checkpoints/tracksense_8class_best.pt --camera 0 --frames 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import YOLO_MODEL_PATH  # noqa: E402
from vision.validate_yolo_checkpoint import check_schema  # noqa: E402

REPORT_DIR = os.path.join("reports", "yolo_smoke")


def _print_detections(result, names):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        print("  (no detections)")
        return
    for box in boxes:
        local_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
        print(f"  {names.get(local_id, local_id):<16} conf={conf:.2f}  bbox=[{x1}, {y1}, {x2}, {y2}]")


def run(model_path: str, image: str = None, camera: int = None, frames: int = 10) -> int:
    if not os.path.exists(model_path):
        print(f"CHECKPOINT MISSING: '{model_path}'. Place the 8-class checkpoint there first.")
        return 2

    from ultralytics import YOLO  # lazy

    model = YOLO(model_path)
    ok, names, checks = check_schema(model.names)
    if not ok:
        print("Refusing to run smoke inference: checkpoint schema is NOT the TrackSense 8-class schema.")
        for label, passed, detail in checks:
            if not passed:
                print(f"  [FAIL] {label}  ({detail})")
        return 1

    os.makedirs(REPORT_DIR, exist_ok=True)

    if image is not None:
        if not os.path.exists(image):
            print(f"Image not found: '{image}'.")
            return 2
        result = model(image, verbose=False)[0]
        print(f"Detections on {image}:")
        _print_detections(result, names)
        out = os.path.join(REPORT_DIR, "smoke_" + os.path.basename(image))
        result.save(filename=out)
        print(f"Annotated image saved -> {out}")
        return 0

    # camera / video frames
    import cv2

    source = 0 if camera is None else camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Could not open camera/video source {source!r}.")
        return 2
    saved = None
    try:
        for i in range(max(1, frames)):
            grabbed, frame = capture.read()
            if not grabbed:
                break
            result = model(frame, verbose=False)[0]
            print(f"Frame {i}:")
            _print_detections(result, names)
            saved = os.path.join(REPORT_DIR, f"smoke_frame_{i:03d}.jpg")
            cv2.imwrite(saved, result.plot())
    finally:
        capture.release()
    if saved:
        print(f"Last annotated frame saved -> {saved}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test YOLO inference on an image or camera.")
    parser.add_argument("--model", default=YOLO_MODEL_PATH)
    parser.add_argument("--image", default=None, help="Path to a single image.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (e.g. 0).")
    parser.add_argument("--frames", type=int, default=10, help="Frames to grab in camera mode.")
    args = parser.parse_args(argv)
    if args.image is None and args.camera is None:
        parser.error("provide --image PATH or --camera INDEX")
    return run(args.model, image=args.image, camera=args.camera, frames=args.frames)


if __name__ == "__main__":
    raise SystemExit(main())
