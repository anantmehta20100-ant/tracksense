# Detection playbook — making the 8-class detector reliable on our camera

Run everything from the repo root on the laptop with `.venv\Scripts\python.exe`
(shortened to `python` below). The class schema stays the 8 classes in
`ml/class_schema.py`; nothing here changes it.

## Where we are

- The live detector, `model/checkpoints/tracksense_8class_multiscene_best.pt`,
  is a **mid-training snapshot**: epoch 4 of a planned 50, saved before the
  Colab run ended. Its own validation mAP50 was still rising every epoch
  (0.708 → 0.737 → 0.757 → 0.812; mAP50-95 0.581).
- The 94.6% mAP50 quoted in `AGENTS.md` belongs to the older
  `tracksense_8class_best.pt`, not this checkpoint.
- Neither number is measured on our camera. Step 1 fixes that.

## 0. Check the GPU is actually used

```
python ml/check_gpu.py
```

It must print `torch.cuda.is_available(): True` and a GPU time far below the
CPU time. The GPU never takes over on its own when the CPU is busy; PyTorch
only uses it when torch has CUDA and the model is put on it. Live inference
and `ml/train_yolo.py` now pick the GPU automatically (`TRACKSENSE_YOLO_DEVICE`,
`--device`, both default `auto`).

## 1. Build the real-camera test set (do this first)

Mount the camera in its demo position. Capture one session for testing only:

```
python ml/capture_frames.py --out data/own_frames/test1
```

Press `a` for auto-capture and keep changing the scene: every class, objects
touching and overlapping, a hand holding cutlery, cluttered and sparse
counters, lights on/off, daylight. 150–300 frames.

Draft the labels with the current model, then correct every box:

```
python ml/prelabel_frames.py --images data/own_frames/test1
```

Open `data/own_frames/test1` in a YOLO-format labelling tool (for example the
X-AnyLabeling Windows release, pointed at `images/` with `classes.txt`). Delete
wrong boxes, add missed ones, tighten loose ones. Never add `counter`.

Score the current model — this is the baseline every later model must beat:

```
python evaluate/evaluate_detector.py --data data/own_frames/test1/data.yaml ^
    --weights model/checkpoints/tracksense_8class_multiscene_best.pt
```

`test1` is never trained on. Capture new sessions for training.

## 2. Finish the interrupted training (GPU, unattended)

Start these while labelling; the trainer keeps Windows awake while it runs
(lid-close still sleeps the laptop, so leave it open and plugged in).

```
REM A: continue the current nano model to a full run
python ml/train_yolo.py --dataset-root data/training_8class_multiscene ^
    --model model/checkpoints/tracksense_8class_multiscene_best.pt --epochs 50 --name ms_finish_n

REM B: same data, the larger "s" model (more accurate, still fast on the GPU)
python ml/train_yolo.py --dataset-root data/training_8class_multiscene ^
    --model yolo26s.pt --epochs 80 --name ms_s
```

Batch 16 at imgsz 640 fits in 8 GB for both; 32 does not. Runs land in
`model/checkpoints/yolo_runs/<name>/weights/best.pt`. `--patience 20` stops a
run early once validation stops improving.

Compare on the real-camera test set:

```
python evaluate/evaluate_detector.py --data data/own_frames/test1/data.yaml --weights ^
    model/checkpoints/tracksense_8class_multiscene_best.pt ^
    model/checkpoints/yolo_runs/ms_finish_n/weights/best.pt ^
    model/checkpoints/yolo_runs/ms_s/weights/best.pt
```

## 3. Teach it our camera

Capture 2–4 more sessions (`data/own_frames/s1`, `s2`, …) on different days
and lighting, aimed at whatever step 2 scored worst. Prelabel and correct
them the same way. Then:

```
python ml/build_owncam_dataset.py --base data/training_8class_multiscene ^
    --train-sessions data/own_frames/s1 data/own_frames/s2 ^
    --test-sessions data/own_frames/test1

python ml/train_yolo.py --dataset-root data/training_8class_owncam ^
    --model <best.pt from step 2> --epochs 40 --name owncam_v1
```

The build warns about label files that are still the untouched model draft —
fix those first, especially in the test session. Score `owncam_v1` against
the step-2 winner with `evaluate_detector.py` on `test1` (or on the combined
set's test split: `--data ml/data.train.resolved.yaml --split test`).

Repeat step 3 for the classes that still score lowest.

## 4. Switch the app to the winner

Copy the winning `best.pt` to `model/checkpoints/tracksense_8class_owncam_best.pt`
and point the app at it, e.g. in `run_tracksense.bat` before Python starts:

```
set TRACKSENSE_YOLO_WEIGHTS=model\checkpoints\tracksense_8class_owncam_best.pt
```

The class-name check refuses any checkpoint that is not the 8-class schema.

## Live settings

All in `config/runtime_config.py`, each overridable by environment variable:

| Setting | Default | Env var |
|---|---|---|
| camera size | 1280×720 (MJPG) | `TRACKSENSE_CAMERA_WIDTH/HEIGHT` |
| inference imgsz | 640 (= training size) | `TRACKSENSE_YOLO_IMGSZ` |
| confidence | 0.25 | `TRACKSENSE_YOLO_CONF` |
| NMS IoU | 0.7 | `TRACKSENSE_YOLO_IOU` |
| device | auto (GPU if visible) | `TRACKSENSE_YOLO_DEVICE` |

The phone page keeps its own `TRACKSENSE_PHONE_IMGSZ` (384).
