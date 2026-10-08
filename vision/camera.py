"""One way to open the webcam, shared by every live path.

DirectShow is the reliable Windows backend. MJPG matters at 1280x720: many
webcams only reach 30 fps at 720p in MJPG and drop to ~5-10 fps in raw YUY2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import CAMERA_HEIGHT, CAMERA_WIDTH  # noqa: E402


def open_camera(source=0, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
    """Open a camera index (configured for live use) or a video file path (as is)."""
    if not isinstance(source, int):
        return cv2.VideoCapture(source)
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(source)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the freshest frame, no backlog
    return cap


def describe(cap) -> str:
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return f"{w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps"
