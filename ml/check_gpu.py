"""Show whether YOLO actually runs on the GPU, and how much faster it is.

    python ml/check_gpu.py [--weights model/checkpoints/tracksense_8class_best.pt]

The laptop GPU never picks up Python work on its own because the CPU is busy:
PyTorch uses it only when torch was installed with CUDA and the model is put on
"cuda". This times the same YOLO call on CPU and on GPU to prove which is used.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import YOLO_IMGSZ, YOLO_MODEL_PATH, resolve_device  # noqa: E402


def time_inference(model, device: str, imgsz: int, runs: int = 20) -> float:
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for _ in range(3):  # warm-up (CUDA init, cudnn autotune)
        model(frame, imgsz=imgsz, device=device, verbose=False)
    start = time.perf_counter()
    for _ in range(runs):
        model(frame, imgsz=imgsz, device=device, verbose=False)
    return (time.perf_counter() - start) / runs * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=YOLO_MODEL_PATH)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    args = parser.parse_args()

    import torch

    print(f"torch {torch.__version__}  built for CUDA {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    elif torch.version.cuda is None:
        print("This torch is a CPU-only build: reinstall a CUDA build of torch to use the GPU.")
    print(f"TrackSense live inference device ('auto' resolves to): {resolve_device()}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    print(f"\n{args.weights} at imgsz {args.imgsz}, 1280x720 frame:")
    cpu_ms = time_inference(model, "cpu", args.imgsz, runs=5)
    print(f"  cpu : {cpu_ms:7.1f} ms/frame  ({1000 / cpu_ms:5.1f} fps)")
    if torch.cuda.is_available():
        gpu_ms = time_inference(model, "0", args.imgsz)
        print(f"  gpu : {gpu_ms:7.1f} ms/frame  ({1000 / gpu_ms:5.1f} fps)  {cpu_ms / gpu_ms:.1f}x faster")
        print(f"  peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
