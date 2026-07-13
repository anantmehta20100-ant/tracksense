# TrackSense

A local AI system that predicts how food-allergen cross-contact risk propagates through a kitchen over time, and warns when a person with a stated allergy may have consumed something contaminated.

See [AGENTS.md](AGENTS.md) for the full project spec, architecture, and 10-day plan. This README only covers setup and run order.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

The pipeline has three stages that must be run in this order: the fine-tuned YOLO weights and the synthetic dataset/trained GRU don't exist until earlier stages produce them.

**1. Fine-tune YOLO on the custom kitchen object classes.** This produces the weights file `ObjectDetector` loads. The dataset pipeline is reproducible (seed 42) and must be run in this order:

```bash
python ml/prepare_hand.py                    # (and the other prepare_*.py, once per class)
python ml/deduplicate_dataset.py             # dry run: SHA-256 exact-duplicate report
python ml/deduplicate_dataset.py --apply     # quarantine duplicates, verify zero leakage
python ml/build_combined_dataset.py          # clean unified dataset, canonical class IDs
python ml/build_balanced_training_dataset.py # balanced TRAIN; valid/test copied verbatim
python ml/perceptual_audit.py --root data/training_8class_balanced --report reports/near_duplicate_review_final.csv
                                             # final advisory pHash review; never auto-deletes
python ml/audit_splits_and_boxes.py          # split quality + strict box sanity
python -m ml.build_manifest                  # reports/training_manifest.csv
python ml/train_yolo.py --smoke              # preflight + dataset-load check (no training)
```

Then train on a Colab T4 from `/content/tracksense`. `--model` is **required**
(no silent default); the project's model family is `yolo26n` (confirmed by
`model/checkpoints/best.pt` train args). Start with batch 32 at 640 pixels:

```bash
python ml/train_yolo.py \
  --model yolo26n.pt \
  --data /content/tracksense/ml/data.8class.colab.yaml \
  --epochs 50 \
  --batch 32 \
  --imgsz 640 \
  --device 0 \
  --workers 8 \
  --seed 42 \
  --name tracksense_8class_t4 \
  --project /content/tracksense/model/checkpoints/yolo_runs
```

If the primary command runs out of memory, retry with the smaller batch command:

```bash
python ml/train_yolo.py \
  --model yolo26n.pt \
  --data /content/tracksense/ml/data.8class.colab.yaml \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --workers 8 \
  --seed 42 \
  --name tracksense_8class_t4_batch16 \
  --project /content/tracksense/model/checkpoints/yolo_runs
```

Training on CPU is refused by default (pass `--force-cpu-train` to override).
Generated reports land in `reports/`.

**2. Generate the synthetic training data:**

```bash
python -m model.synthetic_data --num-sequences 2000 --seed 42
```

Writes `data/synthetic/sequences.csv`.

**3. Train the GRU and compare it against the baselines:**

```bash
python -m model.train
```

Prints an accuracy/F1 comparison table and saves `model/checkpoints/gru.pt`.

**4. Evaluate in more depth (precision/recall/F1, confusion matrix, ranking comparison):**

```bash
python -m evaluate.compare_models
```

**5. Run the live dashboard:**

```bash
streamlit run dashboard/app.py
```

Or, for a debug view without Streamlit:

```bash
python -m pipeline.live_runner --user-allergen nut
```

## Random Forest cross-contact risk model (downstream)

The Random Forest is a **separate, downstream** risk model from the GRU stack above. It does **not** detect objects — YOLO does that. It takes engineered features describing a single contact event plus recent contact history, and predicts a **relative cross-contact risk class** (`LOW` / `MEDIUM` / `HIGH`) for the target object, with class probabilities and a continuous convenience score `P(MEDIUM)·0.5 + P(HIGH)·1.0`.

The feature schema is defined once in `ml/risk_features.py` (the single source of truth for feature order, ranges, and the object vocabulary) and is shared by training and inference, so encoding can never drift.

Run order (independent of the YOLO/GRU steps above):

```bash
# 1. Generate the synthetic development dataset (>= 20k contact events).
#    Labels come from HIDDEN transfer regimes + latent variables that are NOT
#    exposed as features, so the model can't reconstruct one trivial formula.
python ml/generate_risk_training_data.py --num-scenarios 5000 --seed 42
#    -> data/risk_model/risk_events.csv, data/risk_model/scenario_metadata.csv

# 2. Train the Random Forest (grouped split by scenario_id, 70/15/15, seed 42,
#    grouped-CV hyperparameter search, class_weight="balanced").
python ml/train_random_forest.py
#    -> model/risk_random_forest.joblib, model/risk_model_metadata.json

# 3. Evaluate vs baselines (majority / direct-contact rule / logistic regression)
#    on the same held-out test scenarios, with breakdowns and report artifacts.
python evaluate/evaluate_random_forest.py
#    -> reports/risk_model_metrics.json, risk_model_classification_report.txt,
#       risk_model_confusion_matrix.png, risk_model_feature_importance.csv

# 4. Inference API / runtime engine demos.
python ml/risk_inference.py          # predict_contact_risk(event_dict) -> dict
python pipeline/risk_engine.py       # per-object risk state from contact events
```

Tests: `python -m unittest discover -s tests -p "test_*.py"`.

**Runtime integration.** `pipeline/risk_engine.py` (`RiskEngine`) consumes the same `contact_event` dicts that `vision/contact_detector.py` already emits, builds the feature vector, calls the RF, and maintains per-object risk state over time. It is a parallel consumer to the GRU's `pipeline/risk_state.py` and does not modify it. The current proximity/overlap heuristic does not yet measure contact duration, bbox-overlap ratio, or normalized distance, and there is no live cleaning detector — the engine fills those unmeasured feature fields with documented defaults (overridable via `observations=`) rather than fabricating contact events.

## Live end-to-end integration (mock now → YOLO later)

This is the runtime that ties detections → contacts → Random Forest → per-object
risk → dashboard together. It runs **today** on a mock detection source while the
8-class YOLO detector finishes training separately; swapping in the real
`best.pt` later is a one-line config change.

```
Camera / Mock Source        vision/mock_detection_source.py  (or yolo_detection_source.py)
        ↓ detections
Tracking                    vision/tracker.py  (reused IoU tracker, persistent ids)
        ↓ tracked objects
Contact detection           vision/contact_tracker.py  (PENDING→ACTIVE→ENDED, measures
        ↓ ContactEvent       duration / overlap / distance; one event per interaction)
Feature builder             pipeline/risk_feature_builder.py  (→ ml/risk_features schema)
        ↓ features
Random Forest               ml/risk_inference.py  (real inference, model loaded once)
        ↓ RiskPrediction
RiskEngine                  pipeline/risk_engine.py  (per-object risk + propagation chain)
        ↓ risk state + chains
Flask API                   backend/app.py
        ↓ JSON (polling)
React dashboard             backend/static/  (risk list, chains, alerts, timeline, demo controls)
```

- **YOLO detects objects; the Random Forest predicts *relative* downstream risk.** Separate models.
- The contamination **source** for each contact is decided from risk state (a raw allergen class, else the higher-risk object), not geometry — that is what makes the chain flow correctly.
- **Cleaning** is a controlled runtime event (`POST /api/cleaning-event` or `RiskEngine.mark_cleaned`), not a vision guess. The RF predicts the (lower) residual risk; risk is never hard-reset to zero.
- The mock is a *detection* source, not an answer key — it emits boxes/classes over time; all risk is computed downstream by the real RF.

### Run it

```bash
pip install -r requirements.txt            # adds flask

# 1. Launch the backend + dashboard (serves the SPA and the JSON API):
python backend/app.py
#    -> open http://127.0.0.1:8000  (set TRACKSENSE_PORT to change the port)
#    In the dashboard: pick "flagship_chain", click ▶ Start, watch risk propagate
#    nut_butter_jar → cutlery → bread → plate over the timeline.

# 2. Headless demo (no browser) — run the flagship scenario end to end:
python pipeline/demo_controller.py         # prints per-object risk + chains for every scenario
python pipeline/risk_pipeline.py           # flagship only, detailed

# 3. Tests:
python -m unittest discover -s tests -p "test_*.py"
```

The dashboard's React runtime loads from a CDN (unpkg); the backend, pipeline and
model are fully offline. There is no Node/npm build step.

### Plugging in the final 8-class `best.pt`

1. Save the trained 8-class weights to `model/checkpoints/tracksense_8class_best.pt`
   (or set `TRACKSENSE_YOLO_WEIGHTS=/path/to/best.pt`).
2. Set `TRACKSENSE_DETECTION_SOURCE=yolo` and start the backend.
3. On load, `vision/yolo_detection_source.py` **validates the model's class names**
   against the 8-class schema (`ml/class_schema.training_names()`: ids `0..7`,
   `bread==7`, `counter` absent) and **fails fast** on a mismatch — so the old
   single-class `model/checkpoints/best.pt` can never be loaded by accident.

Nothing downstream of the detection source changes: the YOLO adapter emits the
same `Detection` objects the mock does, so tracker → contacts → RF → engine →
dashboard are untouched.

## Scientific honesty / limitations

- **YOLO detects objects; the Random Forest predicts relative cross-contact risk.** They are separate models.
- The Random Forest is currently trained on **synthetic development data** (`ml/generate_risk_training_data.py`), because no public dataset with true allergen-transfer labels exists. Its metrics are **synthetic-development results** — they show it learned the generator's signal, not that it measures real allergen transfer. Every report file states this.
- Random Forest outputs are **relative cross-contact risk**, not biochemical measurements or laboratory-confirmed allergen concentration. Real-world validation would require experimentally measured allergen-transfer data.
- The model predicts **relative risk** based on observed/synthetic interaction patterns, not measured physical allergen concentration. It never claims an exact contamination percentage as ground truth.
- Contact detection uses a **proximity/bounding-box heuristic** between tracked objects, not a trained gesture/grip model. This is a disclosed, deliberate scope decision -- see AGENTS.md's "What changed from the original concept, and why".
- The GRU is trained on **synthetic** contact sequences, since no real labeled cross-contact dataset exists. The live pipeline's *detection* (YOLO + tracker + proximity heuristic) is real-time and real; the *risk-prediction model* was trained offline on synthetic, documented data.
- The live integration currently runs on a **mock detection source** for demo/development; the 8-class YOLO detector is still training separately and its final `best.pt` will replace the mock. The mock emits *detections* (boxes/classes over time), never risk — all risk still comes from the real Random Forest.
- Contact geometry in the mock (overlap/duration) depicts a realistic sustained kitchen contact; the flagship demo's risk values are the model's own outputs, **not hardcoded**. Risk decays with propagation depth, so a downstream object can read MEDIUM/LOW even in a real allergen chain.
- This is a research MVP / proof of concept, not a certified food-safety device.
