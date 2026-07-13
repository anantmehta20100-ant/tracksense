"""Camera diagnostic for physical verification troubleshooting.

Answers "why isn't the webcam giving detections?" by gathering evidence at each
layer instead of guessing: which camera indices open, on which OpenCV backend,
at what resolution, whether frames are black or real image data (mean/std of
pixels), and it saves a raw frame from each working camera so you can SEE what it
captured. Does not touch YOLO -- this isolates the *capture* layer from the
*detection* layer.

Run:
    python vision/diagnose_camera.py
    python vision/diagnose_camera.py --indices 0,1,2 --warmup 10 --out reports/yolo_smoke
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# Windows exposes several capture backends; the default (MSMF) and DSHOW can see
# different device lists, so we probe both to catch a "wrong backend" case.
_BACKENDS = [("default", cv2.CAP_ANY)]
if sys.platform.startswith("win"):
    _BACKENDS += [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)]


def _classify(frame) -> dict:
    """Summarise one frame: is it black / near-uniform / real image data?"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    if mean < 5.0:
        verdict = "BLACK (no light / covered / not delivering video)"
    elif std < 8.0:
        verdict = "NEAR-UNIFORM (blank wall / defocused / covered)"
    else:
        verdict = "REAL IMAGE DATA (camera is capturing a scene)"
    return {"height": frame.shape[0], "width": frame.shape[1],
            "mean_brightness": round(mean, 1), "std": round(std, 1), "verdict": verdict}


def probe(index: int, backend_name: str, backend_id: int, warmup: int, out_dir: str) -> dict:
    cap = cv2.VideoCapture(index, backend_id)
    if not cap.isOpened():
        cap.release()
        return {"index": index, "backend": backend_name, "opened": False}

    # Warm up: the first frames after opening are often black while auto-exposure
    # / auto-focus settle, so we read a few and keep the last good one.
    frame = None
    grabbed_count = 0
    for _ in range(max(1, warmup)):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            grabbed_count += 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    info = {"index": index, "backend": backend_name, "opened": True,
            "frames_grabbed": grabbed_count, "reported_res": f"{width}x{height}",
            "reported_fps": round(fps, 1)}
    if frame is None:
        info["verdict"] = "OPENED but returned NO frames (0 grabbed)"
        return info

    info.update(_classify(frame))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"camdiag_idx{index}_{backend_name}.jpg")
    cv2.imwrite(path, frame)
    info["saved_frame"] = path
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose which camera works and what it sees.")
    parser.add_argument("--indices", default="0,1,2,3", help="Comma-separated indices to probe.")
    parser.add_argument("--warmup", type=int, default=10, help="Frames to grab before sampling.")
    parser.add_argument("--out", default=os.path.join("reports", "yolo_smoke"))
    args = parser.parse_args(argv)

    indices = [int(x) for x in args.indices.split(",") if x.strip() != ""]
    print(f"OpenCV {cv2.__version__} | platform {sys.platform}")
    print(f"Probing camera indices {indices} across backends {[b[0] for b in _BACKENDS]}\n")

    working = []
    for index in indices:
        for backend_name, backend_id in _BACKENDS:
            info = probe(index, backend_name, backend_id, args.warmup, args.out)
            if not info["opened"]:
                print(f"index {index} [{backend_name}]: NOT opened")
                continue
            line = (f"index {index} [{backend_name}]: OPENED  res={info.get('reported_res')} "
                    f"fps={info.get('reported_fps')} grabbed={info.get('frames_grabbed')}")
            if "verdict" in info:
                line += f"\n    -> {info['verdict']}"
            if "mean_brightness" in info:
                line += (f"\n    -> pixels mean={info['mean_brightness']} std={info['std']} "
                         f"({info['width']}x{info['height']})")
            if "saved_frame" in info:
                line += f"\n    -> saved: {info['saved_frame']}"
            print(line)
            if info.get("opened") and info.get("frames_grabbed"):
                working.append(info)

    print("\n=== VERDICT ===")
    real = [w for w in working if "REAL IMAGE" in w.get("verdict", "")]
    if not working:
        print("No camera opened on any index/backend -> hardware/driver/permission issue (not the app).")
    elif not real:
        print("Camera(s) opened but frames are black/uniform -> covered lens, no light, or a "
              "virtual camera. Not a detection bug.")
    else:
        print("Camera IS capturing real image data. If objects still aren't detected, the issue is "
              "FRAMING (objects not in view / too small / poor light), NOT the camera or the code.")
        print("Open the saved camdiag_*.jpg files to see exactly what the camera sees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
