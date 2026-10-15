# Task for the agent on the TrackSense laptop: run the detection playbook

You are working on the user's Windows 11 laptop in the TrackSense repo
(`C:\Users\Anant\Downloads\tracksense`). Read `AGENTS.md` and
`docs/DETECTION_PLAYBOOK.md` first; this file tells you how to run that
playbook for the user and what to report.

**Goal:** make the 8-class YOLO detector reliable on this laptop's own
camera, and measure it honestly.

## Ground rules

- Use `.venv\Scripts\python.exe` for every Python command.
- Work on branch `claude/tender-volta-pz3s0z`. Commit and push code/doc
  changes there only. Never commit `data/`, `datasets/`, `*.pt` weights or
  `model/checkpoints/yolo_runs/`.
- Keep the 8-class schema in `ml/class_schema.py`. Never add `counter`.
- Do not `pip install roboflow` (it breaks `cv2.imshow` and downgrades numpy).
  Do not reinstall or upgrade torch, ultralytics or opencv without asking the
  user first.
- Never print or hardcode API keys.
- Labels on the test set must be corrected by the user. Do not "fix" labels
  yourself to raise a score, and never evaluate on unreviewed drafts and call
  the result a real-camera score.
- Stop and ask the user whenever a step below says **ASK**, or when something
  fails in a way this file does not cover. Quote errors verbatim.

## Step 1 — set up and check (no user needed)

```
git fetch origin
git checkout claude/tender-volta-pz3s0z
git pull
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe ml\check_gpu.py
nvidia-smi
```

- Tests must pass. If not, report and stop.
- `check_gpu.py` must print `torch.cuda.is_available(): True` and a GPU time
  well below the CPU time. If CUDA is not available, report the
  `torch.__version__` / `torch.version.cuda` lines and the `nvidia-smi`
  output, and **ASK** before changing anything.
- Make sure the laptop is on mains power. `ml/train_yolo.py` keeps Windows
  awake while it trains, but closing the lid still sleeps the laptop: tell
  the user to leave it open.

## Step 2 — find the training data (no user needed unless missing)

Training needs `data\training_8class_multiscene\{train,valid,test}\{images,labels}`.

```
.venv\Scripts\python.exe ml\train_yolo.py --dataset-root data\training_8class_multiscene --smoke
```

If the folder is missing, search the disk for a folder named
`training_8class_multiscene` or a zip named like
`tracksense_multiscene_kaggle.zip`. If `data\training_8class_balanced` exists
instead, the docstrings of `ml/build_multiscene_dataset.py` and
`ml/generate_multiscene_yolo_data.py` describe how the multiscene set was
built from it. If none of that exists, **ASK** the user where the dataset is
(it may be on Google Drive or Kaggle).

If the trainer's preflight fails on cross-split duplicate leakage, report
the count and **ASK** before running `ml/deduplicate_dataset.py --apply`.

## Step 3 — train two candidates (unattended, run one after the other)

Run each in the background with its output logged, so you can keep working:

```
.venv\Scripts\python.exe ml\train_yolo.py --dataset-root data\training_8class_multiscene --model model\checkpoints\tracksense_8class_multiscene_best.pt --epochs 50 --name ms_finish_n
.venv\Scripts\python.exe ml\train_yolo.py --dataset-root data\training_8class_multiscene --model yolo26s.pt --epochs 80 --name ms_s
```

- Never run both at once (8 GB of GPU memory). Batch 16 is the default and
  fits; 32 does not.
- After the first epoch of each run, note the seconds per epoch and the GPU
  memory from `nvidia-smi`. If both runs together would take more than about
  8 hours, **ASK** whether to lower `--epochs`.
- Results land in `model\checkpoints\yolo_runs\<name>\weights\best.pt`.

## Step 4 — capture the real-camera test set (user moves objects)

This can run while training. Tell the user exactly this, then start capture:

> The camera is about to record a frame every 2 seconds for 10 minutes.
> Put it in its demo position. Keep changing the scene: the nut butter jar,
> whole nuts, cutlery, a chopping board, plates, bowls and bread; a hand
> holding cutlery; objects touching and overlapping; crowded and sparse
> counters; lights on and off. Every class should appear many times.

```
.venv\Scripts\python.exe ml\capture_frames.py --out data\own_frames\test1 --auto --duration 600
```

Aim for 150–300 frames. If far fewer were saved (static scene), run it again
into the same folder.

## Step 5 — draft labels, then the user corrects them (user needed)

```
.venv\Scripts\python.exe ml\prelabel_frames.py --images data\own_frames\test1
```

Then help the user correct every box in `data\own_frames\test1` (YOLO
format: `images\`, `labels\`, `classes.txt`). Suggest a YOLO-format labelling
tool such as the X-AnyLabeling Windows release.

You may speed this up: look at the drafted frames and give the user a list of
frames with obvious problems (missed objects, wrong class, loose boxes). The
user makes the corrections. Wait until the user says the labels are done.

## Step 6 — score and compare (no user needed)

When both runs have finished and the labels are corrected:

```
.venv\Scripts\python.exe evaluate\evaluate_detector.py --data data\own_frames\test1\data.yaml --weights model\checkpoints\tracksense_8class_multiscene_best.pt model\checkpoints\yolo_runs\ms_finish_n\weights\best.pt model\checkpoints\yolo_runs\ms_s\weights\best.pt
```

Pick the winner: the highest `all` mAP50-95, provided its mAP50 for
`nut_butter_jar`, `cutlery`, `bread` and `plate` (the flagship demo chain) is
not lower than the current model's.

If a new model wins:
1. Copy its `best.pt` to `model\checkpoints\tracksense_8class_v2_best.pt`
   (do not commit it).
2. In `run_tracksense.bat`, add
   `set TRACKSENSE_YOLO_WEIGHTS=model\checkpoints\tracksense_8class_v2_best.pt`
   next to the other `set` lines.
3. Start the app (`run_tracksense.bat`), open the `/camera` page and ask the
   user to confirm boxes look right on the live camera.

If no new model wins, change nothing and report why.

## Step 7 — write up and push

Create `docs/DETECTION_RESULTS.md` with:
- `check_gpu.py` output
- dataset split and class counts printed by the trainer's preflight
- per run: epochs actually run (early stop?), seconds per epoch, GPU memory,
  final validation mAP50 / mAP50-95
- number of test frames and boxes per class in `test1`
- the full `evaluate_detector.py` tables
- which model won and whether the app was switched to it
- every error you hit, verbatim, and how it was resolved
- the 2–3 weakest classes on `test1`, with example frame filenames

Commit `docs/DETECTION_RESULTS.md` and any `run_tracksense.bat` change to
`claude/tender-volta-pz3s0z` and push. Then tell the user, in a few lines,
what improved and what the next round should target (step 3 of
`docs/DETECTION_PLAYBOOK.md`: capture more sessions for the weakest classes
and fine-tune with `ml/build_owncam_dataset.py`).
