# TrackSense

A local AI system that predicts how food-allergen cross-contact risk propagates through a kitchen over time, and warns when a person with a stated allergy may have consumed something contaminated.

**Core insight:** most allergen tools ask "does this object currently contain an allergen?" TrackSense asks "given everything that has touched everything else so far, where could the risk have spread?" A plate that never touched the allergen source directly can still be flagged if it's part of a contact chain (peanut butter → cutlery → bread → plate).

**Hard deadline: 10-day MVP for a program submission.** Scope is deliberately cut down from a larger original concept. See "Guaranteed scope vs stretch goal" below — do not expand scope beyond what's listed there without explicit sign-off, the timeline does not allow it.

---

## Current Project Status

TrackSense is an allergen cross-contact risk propagation prototype.

Current core architecture:

Camera / mock detections
→ YOLO object detection
→ object tracking
→ contact detection
→ Random Forest risk prediction
→ RiskEngine state propagation
→ Flask API
→ React dashboard

The current YOLO detector is an 8-class model trained as:

0 nut_butter_jar
1 whole_nuts
2 hand
3 cutlery
4 chopping_board
5 plate
6 bowl
7 bread

`counter` is not part of the current trained detector and must not be assumed available.

Final YOLO checkpoint:

model/checkpoints/tracksense_8class_best.pt

Do not accidentally load any old single-class `best.pt`.

The Random Forest risk model is a downstream model. It does not detect objects. It predicts relative cross-contact risk from engineered contact-event features and object history.

Risk predictions are prototype relative-risk estimates, not biochemical contamination measurements.

## Current Priority

Do not retrain YOLO.
Do not add Ego4D.
Do not improve the Random Forest unless there is a blocking bug.
Do not return to the GRU as the main architecture.

The priority is one working end-to-end demo:

nut_butter_jar
→ cutlery
→ bread
→ plate

The system should show:

Plate: elevated downstream risk

Reason:
nut_butter_jar → cutlery → bread → plate

## Current Work Plan

1. Ensure `tracksense_8class_best.pt` is stored locally in `model/checkpoints/`.
2. Validate YOLO adapter loads the 8-class model and rejects wrong class schemas.
3. Run camera or video through YOLO and confirm bounding boxes.
4. Add or verify tracking IDs are stable enough for demo.
5. Confirm `contact_tracker.py` emits one meaningful ContactEvent per interaction, not duplicate frame spam. Treat `contact_detector.py` as legacy unless a specific compatibility fix is required.
6. Connect ContactEvent → risk feature builder.
7. Connect feature builder → Random Forest inference.
8. Connect Random Forest → RiskEngine state update.
9. Expose risk state, alerts, and chains through Flask API.
10. Display live risk objects, timeline, and propagation chain in React dashboard.
11. Rehearse and record the flagship demo.
12. Polish pitch and final submission.

> Status: steps 5–10 are already implemented and tested on the mock/RF path (`vision/contact_tracker.py` → `pipeline/risk_feature_builder.py` → `ml/risk_inference.py` → `pipeline/risk_engine.py` → `backend/app.py` → `backend/static/`; 59 tests green). Steps 1–4 are the remaining real-camera wiring that plugs `tracksense_8class_best.pt` in for the mock source; steps 11–12 are rehearsal/pitch.

## Evaluation Focus

For YOLO:
- mAP50
- mAP50-95
- per-class precision and recall
- live detection stability

For risk model:
- accuracy
- balanced accuracy
- macro F1
- confusion matrix
- ranking exposed objects above unaffected objects
- temporal evaluation on held-out scenario groups

Train/test splits for temporal risk sequences must be by `scenario_id` or `sequence_id`, never by individual rows.

Current measured results: **YOLO** — precision 89.4%, recall 91.8%, mAP50 94.6%, mAP50-95 80.1%. **Random Forest** (synthetic-development) — accuracy 82%, macro-F1 0.72, AUC 0.94; beats majority, direct-rule, and logistic-regression baselines.

---

## What changed from the original concept, and why

The original concept assumed training custom hand-object contact detection from video (using EPIC-KITCHENS/VISOR-style datasets). That's cut entirely — reliable hand-gesture/grip understanding from raw video is itself an unsolved research problem, not a buildable module in 10 days.

**What replaced it:** a fine-tuned YOLO object detector (treats "hand" as just one more detected object class, no gesture or grip modeling) plus a simple, explainable proximity heuristic for contact detection — if two tracked objects' bounding boxes overlap or stay within a small distance for N consecutive frames, that's logged as a contact event.

This is an intentional, honest design choice, not a weaker version of the idea: it keeps the project's real ML contribution (the risk-propagation model) as the focus, instead of spending the whole timeline on an unrelated, much harder perception problem. State it plainly in the pitch: "we chose a fast, interpretable heuristic for contact detection so we could focus our effort on the risk-propagation model, which is the actual contribution."

---

## Guaranteed scope vs stretch goal

**Guaranteed deliverable (must work for submission):**
- One allergen category: **nuts** (peanut butter/nut butter jars, whole nuts — tree nuts and peanuts treated as one `allergen_type: "nut"` category for propagation purposes)
- Object detection → tracking → contact events → **Random Forest** risk prediction → RiskEngine → live dashboard, demonstrated end to end
- The flagship propagation demo: `nut_butter_jar → cutlery → bread → plate`, with the plate flagged as downstream risk despite never touching the source

**Stretch goal (only if there is genuinely spare time):**
- A second allergen category, most likely **dairy** (milk carton, butter, cheese — visually easy to detect, reuses the exact same pipeline)

**The one architecture decision that makes the stretch goal cheap:** `allergen_type` must be a first-class parameter/field everywhere — object metadata, synthetic data, model features, dashboard labels — never a hardcoded string like `"nut"` scattered through logic. Adding dairy later should mean "add object classes tagged `allergen_type: dairy`, generate synthetic data, retrain" — not a rewrite. This costs almost nothing to do correctly and should not be skipped even though only nuts are guaranteed.

---

## Architecture (current)

```
   Live camera  OR  mock detection source  (mock is the current default for the demo)
                                 │  detections (boxes, classes, timestamps)
                                 ▼
                      YOLO object detector
             (tracksense_8class_best.pt — 8 classes)
                                 │
                                 ▼
                        Object tracker
             (IoU, consistent track IDs across frames)
                                 │
                                 ▼
                   Contact detector / tracker
        (proximity + overlap, N-frame persistence + hysteresis;
         ONE meaningful ContactEvent per interaction)
                                 │
                                 ▼
                 Random Forest risk prediction
   (engineered contact-event features + object history →
    LOW/MEDIUM/HIGH + probabilities + relative risk score)
                                 │
                                 ▼
                          RiskEngine
    (per-object risk state + propagation chain: parent /
     root allergen / full risk_chain of track ids)
                                 │
                                 ▼
                          Flask API
   (risk map, alerts, propagation chains, timeline; JSON polling;
    demo controls; POST /api/cleaning-event)
                                 │
                                 ▼
                        React dashboard
     (live risk objects, timeline, propagation chain, alerts)
```

**Fully local.** No cloud API. Camera/mock → YOLO → tracking → contact detection → Random Forest → RiskEngine → Flask → React, all on the user's machine.

**Key behaviours:**
- The contamination **source** of each contact is chosen from risk state (a raw allergen class, else the higher-risk object), not geometry — this is what makes the chain flow in the right direction.
- **Cleaning** is a controlled runtime event (`POST /api/cleaning-event` / `RiskEngine.mark_cleaned`); the RF predicts the lower residual risk (no hard reset to zero).
- The `flagship_chain` demo produces `nut_butter_jar → cutlery → bread → plate` with the plate flagged despite no direct contact — **all risk values are the RF's own outputs, not hardcoded.**
- Run it: `python backend/app.py` → open `http://127.0.0.1:8000`, pick a scenario, ▶ Start. Tests: `python -m unittest discover -s tests -p "test_*.py"`.

**Legacy GRU path (kept, NOT primary).** An earlier GRU temporal model plus a MediaPipe mouth-proximity "consumption/exposure alert" feed a separate **Streamlit** dashboard (`model/`, `pipeline/risk_state.py`, `pipeline/live_runner.py`, `pipeline/consumption.py`, `dashboard/app.py`). It still runs and must not be deleted, but it is not the current architecture — do not re-center the project on it.

---

## The risk-propagation model (current: Random Forest)

The risk model does not classify a single frame. It scores each **contact event** using that event's features plus recent interaction history, so the prediction for the current contact depends on what came before it — not the contact in isolation.

**Current model — Random Forest** (`ml/`), complete and used by the live pipeline. It does **not** detect objects (YOLO does). Input per event (15 features, order fixed once in `ml/risk_features.py`): source/target object class, current source risk, is-source-allergen, measured contact geometry (duration, overlap, distance), time since last contact, propagation depth (hops from the original source), repeated-contact count, seconds since source exposure, cleaning flag. Output: relative risk class LOW/MEDIUM/HIGH + probabilities + a continuous relative-risk score.

**Development benchmark (synthetic, held-out scenario groups):** accuracy 82%, macro-F1 0.72, AUC 0.94 — beats majority, direct-contact-rule, and logistic-regression baselines (`evaluate/evaluate_random_forest.py`, `reports/risk_model_*`). These are synthetic-development results, **not** measured contamination.

**Legacy GRU alternative (kept, not primary).** An earlier GRU temporal-sequence model (`model/gru_model.py`, `model/train.py`) carried interaction history in a recurrent hidden state and was compared against direct-contact / fixed-decay / non-temporal baselines. Retained for reference; not the current architecture and not the central claim.

---

## Dataset strategy — synthetic, not real video-derived labels

There is no existing dataset with ground-truth labels like "this plate has 73% nut cross-contact risk after this exact chain." Do not source or hand-label one from real video within this timeline.

**Current — the RF synthetic generator** (`ml/generate_risk_training_data.py`) produces randomized, labeled contact-event scenarios (allergen source → A → B, branches, repeats, cleaning). Labels come from **hidden latent transfer regimes** that are never exposed as features, plus noise — so the model learns real signal instead of reconstructing one deterministic formula. Grouped by `scenario_id` for leakage-free splits (70/15/15, seed 42). Outputs land in `data/risk_model/`.

**Disclosure principle (state clearly in the pitch):** the perception layer (YOLO + tracker + proximity heuristic) is live and real; the risk model was trained offline on a synthetic, documented dataset, since no real cross-contact ground-truth dataset exists — disclosed, not hidden. `allergen_type` stays a first-class field so a second allergen (dairy) is a config addition, not a rewrite.

*(Legacy: the GRU path used a separate generator `model/synthetic_data.py` → `data/synthetic/sequences.csv`, schema `sequence_id, event_index, source_object, target_object, allergen_type, source_risk, contact_type, time_since_previous_contact, cleaning_event, propagation_depth, target_risk_label`. Retained for the GRU stack.)*

---

## Object classes

**Canonical registry — 9 classes (`config/allergens.py`):**

```
Sources (allergen_type: nut):  nut_butter_jar, whole_nuts
Utensils:                      hand, cutlery, chopping_board
Surfaces:                      plate, bowl, counter
Food:                          bread
```

**The TRAINED detector is 8 classes, NOT 9.** `counter` has no training data and is **excluded** from the trained model — do not assume it is available, and never fabricate `counter` detections. Trained model-local ids (`ml/class_schema.py`, `tracksense_8class_best.pt`):

```
0 nut_butter_jar   4 chopping_board
1 whole_nuts       5 plate
2 hand             6 bowl
3 cutlery          7 bread   ← bread is local id 7 (slides into counter's empty slot)
```

Runtime code consuming model output must map model-local ids back to canonical ids via `ml/class_schema.model_to_canonical()` before touching `config/allergens.py`. Do not accidentally load an old single-class `best.pt` — the YOLO adapter (`vision/yolo_detection_source.py`) validates class names on load and rejects a mismatched schema.

**Trained YOLO validation (8-class):** precision 89.4%, recall 91.8%, mAP50 94.6%, mAP50-95 80.1%.

**Cutlery is single-class** (the Roboflow dataset had no knife/fork/spoon split — 9,038 instances, one class id). Collapsed into one `cutlery` class; fine, because risk propagation depends on detecting *that* a contact occurred, not the utensil type.

`hand` carries and spreads contamination like any other object — no gesture/grip modeling, just one more detected class. **FOOD_CLASSES** for the legacy consumption check (`bread`, `whole_nuts`, `nut_butter_jar`) are defined in `config/allergens.py`.

---

## Consumption / exposure alert feature (legacy GRU / Streamlit path)

*This belongs to the legacy GRU/Streamlit path and is secondary to the current RF integration; kept, not primary.*

Before monitoring starts, the user selects their allergy (`allergen_type`, e.g. "nut"), stored as the session's `user_allergen`. During monitoring, MediaPipe Face Detection locates the mouth region; if a food object with current risk (for the matching `allergen_type`) stays near the mouth for N consecutive frames, a **consumption event** `{object_id, allergen_type, risk_at_time, timestamp}` is logged. If `risk_at_time` exceeds the threshold, a distinct **exposure alert** fires. This is a heuristic layer consuming the risk model's output, not itself a trained model.

---

## Scientific honesty / limitations (must appear in the pitch, not just here)

- The risk model predicts **relative risk** from observed/synthetic interaction patterns, not measured physical allergen concentration. Never claim an exact contamination percentage as ground truth.
- Contact detection uses **proximity/bounding-box heuristics**, not a trained gesture/grip model. Disclosed, deliberate scope decision.
- The risk models (current Random Forest; legacy GRU) are trained on **synthetic** contact data, since no real labeled cross-contact dataset exists. The live pipeline's *detection* is real; the *risk prediction* is a model trained offline on synthetic, documented data.
- This is a research MVP / proof of concept, not a certified food-safety device.

---

## Known limitations in the current scaffold

**Cleaning is a manual runtime event, not vision-detected.** The RF path supports cleaning via a controlled event (`POST /api/cleaning-event` / `RiskEngine.mark_cleaned`) that lowers an object's risk to the RF's predicted residual (no hard reset). But vision cannot yet *detect* cleaning — no trained class is a sponge/cloth/sink — so a real cleaning action must be signalled manually. Cheapest future fix: add a `cleaning_tool` class + a contact rule. (The legacy GRU path passes `cleaning_event=False` for all live events.)

**Polling latency.** The dashboard polls JSON endpoints (e.g. `/api/snapshot`, ~400 ms) rather than receiving push updates, so there is a small bounded lag between an event and the screen. Acceptable for this use case; the documented upgrade path is `flask-socketio` if it ever feels sluggish, not faster polling.

---

## Project structure

Actual repo layout (✅ built · ⏳ planned/not built). GRU path and RF/integration path are marked.

```
tracksense/
├── AGENTS.md · README.md · requirements.txt
├── config/
│   ├── allergens.py            # ✅ canonical 9-class + allergen_type registry, thresholds
│   └── runtime_config.py       # ✅ RF integration: detection source, ports, contact/alert/demo knobs
├── data/
│   ├── synthetic/              # legacy GRU sequences (gitignored)
│   ├── risk_model/             # ✅ RF synthetic events + scenario metadata (gitignored)
│   └── training_photos/        # custom YOLO photos (gitignored)
├── vision/
│   ├── object_detector.py      # ✅ YOLO wrapper
│   ├── face_detector.py        # ✅ MediaPipe face (legacy GRU consumption path)
│   ├── tracker.py              # ✅ IoU tracker, persistent ids (reused by both paths)
│   ├── contact_detector.py     # ✅ legacy GRU path: minimal proximity contact events
│   ├── contact_tracker.py      # ✅ RF path: lifecycle contacts w/ measured geometry
│   ├── mock_detection_source.py# ✅ RF path: 5 scripted scenarios (stands in for the camera now)
│   └── yolo_detection_source.py# ✅ RF path: 8-class adapter (validates names, rejects wrong schema)
├── ml/
│   ├── class_schema.py         # ✅ canonical<->model-local 8-class mapping (counter excluded, bread→7)
│   ├── (YOLO dataset build/dedup/audit/train scripts, data.*.yaml)  # ✅
│   ├── risk_features.py        # ✅ RF: single source of feature order/schema
│   ├── generate_risk_training_data.py # ✅ RF: hidden-latent synthetic generator
│   ├── train_random_forest.py  # ✅ RF: grouped split + train, saves joblib + metadata
│   ├── risk_baselines.py       # ✅ RF: majority / direct-rule / logistic baselines
│   └── risk_inference.py       # ✅ RF: predict_contact_risk(), model loaded once
├── model/
│   ├── synthetic_data.py · gru_model.py · baselines.py · train.py   # ✅ legacy GRU stack
│   ├── checkpoints/tracksense_8class_best.pt  # final 8-class detector (store here; gitignored)
│   ├── checkpoints/best.pt     # ⚠ old checkpoint — do NOT load as the 8-class detector
│   ├── risk_random_forest.joblib # ✅ trained RF (gitignored)
│   └── risk_model_metadata.json  # ✅ RF metadata
├── pipeline/
│   ├── risk_state.py · consumption.py · live_runner.py  # ✅ legacy GRU live path
│   ├── contracts.py            # ✅ RF path: Detection/ContactEvent/ObjectRiskState/RiskPrediction
│   ├── risk_feature_builder.py # ✅ RF path: contact + state -> feature schema
│   ├── risk_engine.py          # ✅ RF path: per-object risk + propagation-chain provenance
│   ├── propagation.py          # ✅ RF path: chain explanations ("jar → cutlery → bread → plate")
│   ├── risk_pipeline.py        # ✅ RF path: source→tracker→contact→RF→engine spine
│   └── demo_controller.py      # ✅ RF path: scenario playback for the live demo
├── dashboard/
│   └── app.py                  # ✅ legacy GRU dashboard (Streamlit)
├── backend/
│   ├── app.py                  # ✅ RF path: Flask API + serves the SPA (status/objects/risk-map/
│   │                           #    events/alerts/snapshot, demo controls, cleaning-event)
│   └── static/                 # ✅ RF path: index.html + app.js (React via CDN, polling; no build)
├── frontend/                   # ⏳ planned Vite React app — NOT built (backend/static/ delivered instead)
├── evaluate/
│   ├── compare_models.py       # ✅ legacy GRU vs baselines
│   └── evaluate_random_forest.py # ✅ RF vs baselines, reports/risk_model_*
└── tests/
    ├── test_perceptual_audit.py · test_risk_model.py            # ✅
    └── test_integration.py · test_backend_api.py               # ✅ live integration + Flask API (59 total, green)
```

**Delivered dashboard:** `backend/app.py` (Flask) serves a React-via-CDN SPA (`backend/static/`) with JSON polling — runs with `python backend/app.py`, no npm/Vite build, same-origin so no CORS. **Optional future upgrades (not required for the demo):** an MJPEG `/video_feed` route for live camera frames, migrating to a Vite `frontend/`, `flask-socketio` push updates, and the shared DM-Serif/paper-ink design system used in the developer's other projects (ResumeAlign, TalkCoach, PostureAI). The current SPA uses a neutral palette.

---

## Evaluation

**Current risk model (Random Forest):** accuracy, balanced accuracy, macro-F1, confusion matrix; ranking exposed objects above unaffected ones; per-scenario-group temporal behaviour. Compared against majority / direct-contact-rule / logistic-regression baselines on held-out scenarios (`evaluate/evaluate_random_forest.py`, `reports/risk_model_*`). Benchmark: accuracy 82%, macro-F1 0.72, AUC 0.94.

**YOLO detector:** mAP50, mAP50-95, per-class precision/recall, live detection stability. Current: P 89.4%, R 91.8%, mAP50 94.6%, mAP50-95 80.1%.

**Split methodology — by `scenario_id` / `sequence_id`, never by row.** Rows within one scenario are sequentially dependent (features reference `propagation_depth` and history from earlier events in the same chain), so a random row-level split would leak between train and test. Hold out entire scenarios/sequences.

*(Legacy: the GRU-vs-baselines temporal comparison lives in `evaluate/compare_models.py`; it is no longer the central project claim.)*

---

## Original 10-day plan (historical — superseded by "Current Work Plan")

The original day-by-day plan (fine-tune YOLO → build synthetic data → build the risk model + baselines → wire the live pipeline → build the Flask backend → build the dashboard → rehearse) is largely **complete**: the 8-class YOLO is trained, the Random Forest and the full live integration are built, and the Flask + React dashboard runs. Remaining work and priorities are in **Current Work Plan** and **Current Priority** at the top of this file. Nuts remain the guaranteed scope; dairy is the only stretch goal and the first thing to cut under time pressure — a working, rehearsed nuts-only demo beats a rushed two-allergen demo that breaks live.

---

## Conventions

- Stripped-down, readable code; imports at the top of files; minimal comments except where clarifying non-obvious math or heuristic thresholds.
- Prefer explicit, readable heuristics over cleverness — the project's credibility depends on being able to explain every design decision plainly to a non-technical judge or a doctor.
- Every configurable threshold belongs in `config/`, not hardcoded in logic files.
- `allergen_type` should remain parameterized where relevant, never a hardcoded string baked into control flow.
- Keep YOLO, contact detection, risk prediction, and dashboard concerns separated.
- Do not delete the GRU stack, but do not treat it as the primary architecture.
- Preserve the 8-class local YOLO schema; never fabricate `counter` detections.
- Never claim the RF model measures real allergen concentration — it predicts relative cross-contact risk only.
- Do not use Ego4D during the current deadline unless explicitly instructed later.
