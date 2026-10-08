"""Grab frames from the real TrackSense webcam for labelling.

Frames from the actual camera, angle, lighting and counter are worth far more
than any downloaded dataset: they become the honest test set, and later the
best training data.

    python ml/capture_frames.py --camera 0 --out data/own_frames/session1

Keys in the preview window:
  space  save this frame now
  a      toggle auto-capture (one frame every --every seconds, if the scene changed)
  q      quit

Saved frames are the raw camera image (no overlay), JPEG quality 95.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.camera import describe, open_camera  # noqa: E402


def thumb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", default=f"data/own_frames/{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--every", type=float, default=2.0, help="auto-capture interval (s)")
    parser.add_argument("--min-change", type=float, default=6.0,
                        help="auto-capture skips frames whose mean grey-level change vs the last save is below this")
    parser.add_argument("--max", type=int, default=1000)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    print(f"camera {args.camera}: {describe(cap)} -> saving to {out.resolve()}")

    auto, saved, last_save, last_thumb = False, len(list(out.glob("*.jpg"))), 0.0, None
    window = "capture  [space] save  [a] auto  [q] quit"
    while saved < args.max:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        now = time.time()
        save = False
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            auto = not auto
        if key == ord(" "):
            save = True
        elif auto and now - last_save >= args.every:
            t = thumb(frame)
            save = last_thumb is None or float(np.abs(t - last_thumb).mean()) >= args.min_change
            if not save:
                last_save = now  # unchanged scene: wait another interval
        if save:
            path = out / f"frame_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved, last_save, last_thumb = saved + 1, now, thumb(frame)

        view = frame.copy()
        status = f"{frame.shape[1]}x{frame.shape[0]}  saved {saved}  auto {'ON' if auto else 'off'}"
        cv2.putText(view, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(view, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(window, view)

    cap.release()
    cv2.destroyAllWindows()
    print(f"{saved} frames in {out.resolve()}")


if __name__ == "__main__":
    main()
