# YOLO Multiscene Checkpoint — Recovery & Validation Status

_Last updated: 2026-07-12_

## Final model path

```
model/checkpoints/tracksense_8class_multiscene_best.pt
```

- Size: **20.79 MB** (20,785,154 bytes)
- Not renamed — the `multiscene` name is kept deliberately to distinguish it from the
  older `model/checkpoints/tracksense_8class_best.pt` and the single-class
  `model/checkpoints/best.pt`.
- This path is git-ignored (`model/checkpoints/` in `.gitignore`), so the weights
  are **not** committed — only source/docs are.

## Source

Recovered from Google Drive:

```
tracksense_8class_multiscene_FINAL_MODEL.zip
  ├── tracksense_8class_multiscene_best.pt   (the checkpoint)
  └── model_metadata.json
```

`model_metadata.json` (from the zip):

- `model`: `tracksense_8class_multiscene_best.pt`
- `source`: "Colab full run best.pt"
- `saved_at`: 2026-07-13 01:51:07
- `physical_verification`: **false**

## Validation status

**SCHEMA VALID.** `vision/validate_yolo_checkpoint.py` loaded the checkpoint with
ultralytics and every sub-check passed:

- [PASS] exactly 8 classes (not 9)
- [PASS] `counter` absent
- [PASS] `bread` is local id 7
- [PASS] not a single-class model (rejects the old cutlery `best.pt`)
- [PASS] exact schema match (ids + names)

Command:

```
python vision/validate_yolo_checkpoint.py --model model/checkpoints/tracksense_8class_multiscene_best.pt
```

## Schema

```
0 nut_butter_jar
1 whole_nuts
2 hand
3 cutlery
4 chopping_board
5 plate
6 bowl
7 bread
```

No `counter` class. `bread` is id 7. 8 classes exactly.

## Import fix (ModuleNotFoundError: No module named 'pipeline')

No new file was created. `pipeline/contracts.py` **already exists** and already
defines the `Detection` dataclass (fields: `track_id`, `class_id`, `class_name`,
`confidence`, `bbox_xyxy`, `frame_index`, `timestamp`) plus `ContactEvent`,
`RiskPrediction`, and `ObjectRiskState`. The live detection module
`vision/yolo_detection_source.py` — and the validation/smoke scripts — already
insert the project root onto `sys.path`
(`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`), so
`from pipeline.contracts import Detection` resolves cleanly when run from the
project root. Creating a second `contracts.py` would have duplicated `Detection`
and `ContactEvent`, so it was deliberately avoided. Verified:

```
python -c "from pipeline.contracts import Detection; print(Detection.__name__)"   # OK
python -c "from vision.yolo_detection_source import YoloDetectionSource"           # OK
```

## Smoke test (still-image, no camera)

Ran real still-image inference on the two local peanut-butter + cutlery photos in
`reports/yolo_smoke/` (no invented or internet images):

| Image | Detections |
|---|---|
| `peanut_cutlery_A.jpg` | `cutlery` conf 0.93 (+ 2 low-conf cutlery 0.32 / 0.26) |
| `peanut_cutlery_B.jpg` | `nut_butter_jar` conf 0.88 |

Annotated outputs saved to `reports/yolo_smoke/smoke_peanut_cutlery_A.jpg` and
`…_B.jpg`.

Honest read: the model detects **cutlery** and **nut_butter_jar** individually at
high confidence, but neither of these two frames shows both objects co-detected in
a single image. Single-frame multi-object co-detection (the whole point of the
multiscene retrain) is **not yet demonstrated on real photos** — that needs a real
image containing a jar and cutlery together.

## Test + demo status at recovery time

- `python -m unittest discover -s tests -p "test_*.py"` → **Ran 81 tests, OK** (all pass).
- `python pipeline/headless_demo_runner.py --scenario flagship_chain --frames 150` →
  `FLAGSHIP SUCCESS: True`, chain `nut_butter_jar → cutlery → bread → plate`,
  `physical_verification=False`. Model placement did not break the mock/headless demo.

## Limitations

- `physical_verification=false` — carried through from the model metadata and the
  headless demo. **No physical/biochemical verification has been performed.**
- **No live camera verification yet.**
- The model predicts **objects only**, not biochemical allergen presence. A detected
  jar/cutlery is a proxy for possible allergen contact, not a measurement of allergen.
- Real-image co-detection of multiple objects in one frame is not yet demonstrated
  (see smoke test above).
- Risk math / thresholds were **not** changed during recovery.

## Next steps

1. **Still-image smoke test** with a real photo that contains a peanut-butter jar
   **and** cutlery together, to confirm single-frame co-detection on real data.
2. **Live camera test** (`vision/smoke_yolo_inference.py --camera 0 …`) once a
   co-detection still image looks right.
3. **Dashboard / live-pipeline integration** with the multiscene weights if stable —
   the live path (`TRACKSENSE_DETECTION_SOURCE=yolo`) reads
   `config.runtime_config.YOLO_MODEL_PATH`, which currently defaults to
   `model/checkpoints/tracksense_8class_best.pt`. To point the live pipeline at the
   recovered multiscene model, either set
   `TRACKSENSE_YOLO_WEIGHTS=model/checkpoints/tracksense_8class_multiscene_best.pt`
   or update that default. (Left unchanged for now — validation/smoke used the
   explicit `--model` flag; risk math untouched.)
