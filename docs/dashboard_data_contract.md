# Dashboard data contract

The JSON the frontend consumes, so UI work can proceed **without a live camera**.
Every field below is produced today by the mock/replay + RF pipeline and served
by the Flask backend (`backend/app.py`) and written by the headless runner
(`pipeline/headless_demo_runner.py`). Source of truth for the shape:
`pipeline/risk_pipeline.py::RiskPipeline.snapshot()` (+ demo fields added by
`pipeline/demo_controller.py`).

All risk numbers are the Random Forest's own relative-risk outputs — never
hardcoded, never a measured contamination level.

## Endpoints (Flask, JSON polling, same-origin)

| Method | Path | Returns |
|---|---|---|
| GET  | `/api/status`        | config + demo status (source, scenario, cursor, progress, scenarios) |
| GET  | `/api/snapshot`      | everything below in one poll (the primary UI feed) |
| GET  | `/api/objects`       | `{objects: [ObjectRisk]}` |
| GET  | `/api/risk-map`      | `{timestamp, objects: [ObjectRisk]}` |
| GET  | `/api/events`        | `{events: [TimelineEntry]}` |
| GET  | `/api/alerts`        | `{alerts: [Alert]}` |
| GET  | `/api/explanations`  | `{explanations: [Explanation]}` |
| POST | `/api/demo/start`    | `{scenario?, speed?}` → snapshot; begins auto-play |
| POST | `/api/demo/pause`    | snapshot |
| POST | `/api/demo/resume`   | snapshot |
| POST | `/api/demo/step`     | advance one frame → snapshot |
| POST | `/api/demo/run`      | run to completion → snapshot |
| POST | `/api/demo/reset`    | `{scenario?}` → snapshot |
| POST | `/api/cleaning-event`| `{track_id?｜class?}` → `{ok, track_id, prediction, snapshot}` |

## Snapshot shape (`GET /api/snapshot`)

```jsonc
{
  "source_kind": "mock",                 // "mock" | "yolo"
  "source": "mock",                      // alias under the contract field name
  "model": "tracksense_8class_best.pt",  // configured 8-class detector (basename)
  "physical_verification": false,        // HONESTY FLAG — see below
  "frame_index": 129,
  "timestamp": 12.9,
  "scenario": "flagship_chain",          // added by the demo controller / headless runner
  "status": "finished",                  // idle | running | finished (demo controller)
  "cursor": 130, "total_frames": 130, "progress": 1.0,   // demo controller
  "tracked_objects": [ { "track_id": 0, "class_name": "nut_butter_jar", "bbox_xyxy": [..4..] } ],
  "objects":       [ ObjectRisk, ... ],  // per-object risk state, risk desc
  "explanations":  [ Explanation, ... ], // why each downstream object is elevated
  "alerts":        [ Alert, ... ],       // objects crossing the alert threshold
  "active_contacts": [ { "pair": [0,1], "phase": "active", "close_streak": 6 } ],
  "timeline":      [ TimelineEntry, ... ]
}
```

### ObjectRisk (`objects[]`)

```jsonc
{
  "track_id": 3,
  "class_name": "plate",
  "risk_score": 0.384,          // continuous relative score [0,1]
  "risk_class": "MEDIUM",       // LOW | MEDIUM | HIGH (RF-authoritative)
  "risk_class_id": 1,
  "contact_count": 1,
  "propagation_depth": 2,       // hops from the root allergen source
  "probabilities": { "LOW": .., "MEDIUM": .., "HIGH": .. },
  "parent_track_id": 2,
  "root_allergen_track_id": 0,
  "risk_chain": [0, 1, 2, 3],   // track_ids: root allergen -> ... -> this object
  "is_allergen_source": false,
  "last_updated": 11.4,
  "last_cleaned_time": null
}
```

To render a chain of class names, map each `risk_chain` track_id through
`objects[]` (`{track_id -> class_name}`), e.g. `[0,1,2,3]` →
`nut_butter_jar → cutlery → bread → plate`.

### Explanation (`explanations[]`) and Alert (`alerts[]`)

```jsonc
// Explanation — every elevated (MEDIUM+) downstream object with a real chain
{
  "object": "plate", "track_id": 3, "risk_class": "MEDIUM", "risk_score": 0.384,
  "propagation_depth": 2, "root_allergen_track_id": 0,
  "chain": [ {"track_id":0,"class":"nut_butter_jar"}, ... {"track_id":3,"class":"plate"} ],
  "chain_text": "nut_butter_jar → cutlery → bread → plate",
  "note": "Predicted downstream cross-contact risk (relative, not a measured contamination level)."
}
// Alert — subset of explanations that cross the alert threshold (HIGH class or score >= 0.6)
{ "object":"cutlery", "track_id":1, "risk_class":"HIGH", "risk_score":0.707,
  "chain_text":"nut_butter_jar → cutlery", "chain":[ ... ] }
```

### TimelineEntry (`timeline[]` / `events[]`)

One of `type`: `source_detected`, `contact`, `cleaning`.

```jsonc
{ "type":"contact", "timestamp":3.8, "frame_index":38, "event_id":1,
  "source_class":"nut_butter_jar", "target_class":"cutlery",
  "source_track_id":0, "target_track_id":1,
  "duration":2.2, "overlap_ratio":0.4006, "normalized_distance":0.0,
  "risk_class":"HIGH", "risk_score":0.7072 }
```

## Source mode & physical-verification status (must surface in the UI)

- **`source` / `source_kind`**: `mock` (deterministic replay — the current demo
  default; the YOLO detector is NOT invoked), `yolo` (live camera/video through
  `tracksense_8class_best.pt`). `video` / `camera` are yolo sub-modes when live
  wiring lands.
- **`model`**: the 8-class detector the pipeline is configured for. Present even
  in mock mode (documents the target detector); `source=mock` disambiguates that
  no YOLO inference ran this session.
- **`physical_verification`**: always `false` today. The UI should show a small
  "replay / not physically verified" badge while this is false, and must never
  present risk as a measured contamination level.

## Headless artifacts (same contract, file form)

`python pipeline/headless_demo_runner.py --scenario flagship_chain --frames 150`
writes `reports/headless_demo/`:
- `events.jsonl` — one row per frame (`frame_index`, `timestamp`, `detections`,
  `tracked_objects`, `contacts`, `risk_updates`, `elevated_objects`, `explanations`).
- `final_snapshot.json` — the snapshot shape above (+ `frames_processed`).
- `summary.txt` — human-readable run + success-criteria summary.
