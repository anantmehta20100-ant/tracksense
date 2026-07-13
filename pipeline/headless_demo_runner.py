"""Headless (no-camera) end-to-end demo runner.

Drives the REAL RF pipeline from the deterministic mock replay source, with no
camera and no physical assumptions, and writes dashboard-ready artifacts:

    reports/headless_demo/events.jsonl        one row per processed frame
    reports/headless_demo/final_snapshot.json the pipeline snapshot at the end
    reports/headless_demo/summary.txt         human-readable run summary

The flow is exactly the live path, source-swapped to the mock replay -- nothing
here fabricates risk or bypasses the contact tracker / RF / RiskEngine:

    MockDetectionSource.frames()        (same FrameData the YOLO adapter emits)
      -> RiskPipeline.process_frame()   (IoUTracker -> ContactTracker -> RF -> RiskEngine)
      -> per-frame + final snapshot     (objects / contacts / chains / alerts)

Every risk number comes from the Random Forest via RiskPipeline; this script only
observes and records. It marks physical_verification=false everywhere: this is
software/replay integration, NOT real-world contact accuracy.

CLI:
    python pipeline/headless_demo_runner.py --scenario flagship_chain --frames 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import DEMO, RF_MODEL_PATH, YOLO_MODEL_PATH  # noqa: E402
from pipeline.propagation import build_alerts, build_explanations  # noqa: E402
from pipeline.risk_pipeline import RiskPipeline  # noqa: E402
from vision.mock_detection_source import MockDetectionSource, list_scenarios  # noqa: E402

DEFAULT_OUT_DIR = os.path.join("reports", "headless_demo")

# The flagship contact chain the demo must reproduce (source -> ... -> plate).
FLAGSHIP_PAIRS = [
    ("nut_butter_jar", "cutlery"),
    ("cutlery", "bread"),
    ("bread", "plate"),
]


def _elevated(risk_map) -> list:
    """Objects the engine currently rates above LOW, highest risk first."""
    out = [o for o in risk_map.values() if o["risk_class"] != "LOW"]
    out.sort(key=lambda o: o["risk_score"], reverse=True)
    return [{"class_name": o["class_name"], "track_id": o["track_id"],
             "risk_class": o["risk_class"], "risk_score": o["risk_score"]} for o in out]


def _unordered_pair(a: str, b: str):
    return frozenset((a, b))


def run(scenario: str = "flagship_chain", frames: int = None, *, model_path=None,
        fps: float = None, seed: int = None, out_dir: str = DEFAULT_OUT_DIR) -> dict:
    """Run the headless demo and write the three report artifacts. Returns a
    dict of run metadata + the success-criteria check results."""
    if scenario not in list_scenarios():
        raise ValueError(f"unknown scenario {scenario!r}; choose from {list_scenarios()}")

    fps = float(fps if fps is not None else DEMO.fps)
    seed = int(seed if seed is not None else DEMO.seed)
    model_path = model_path if model_path is not None else RF_MODEL_PATH

    pipeline = RiskPipeline(model_path=model_path, source_kind="mock")
    source = MockDetectionSource(scenario, fps=fps, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    events_path = os.path.join(out_dir, "events.jsonl")
    snapshot_path = os.path.join(out_dir, "final_snapshot.json")
    summary_path = os.path.join(out_dir, "summary.txt")

    contacts_seen = []          # every contact timeline entry, in order
    processed = 0

    with open(events_path, "w", encoding="utf-8") as handle:
        for fd in source.frames():
            if frames is not None and processed >= frames:
                break

            timeline_before = len(pipeline.timeline)
            predictions = pipeline.process_frame(fd)
            new_timeline = pipeline.timeline[timeline_before:]

            frame_contacts = [e for e in new_timeline if e["type"] == "contact"]
            contacts_seen.extend(frame_contacts)
            risk_map = pipeline.engine.risk_map()

            row = {
                "frame_index": fd.frame_index,
                "timestamp": round(fd.timestamp, 3),
                "detections": sorted(t.class_name for t in pipeline.tracks),
                "tracked_objects": [
                    {"track_id": t.track_id, "class_name": t.class_name} for t in pipeline.tracks
                ],
                "contacts": [
                    {"source_class": c["source_class"], "target_class": c["target_class"],
                     "source_track_id": c["source_track_id"], "target_track_id": c["target_track_id"],
                     "duration": c["duration"], "overlap_ratio": c["overlap_ratio"],
                     "risk_class": c["risk_class"], "risk_score": c["risk_score"]}
                    for c in frame_contacts
                ],
                "risk_updates": [
                    {"track_id": p["target_track_id"], "risk_class": p["risk_class"],
                     "risk_score": p["risk_score"]}
                    for p in predictions if isinstance(p, dict) and "target_track_id" in p
                ],
                "control_events": list(fd.control_events),
                "elevated_objects": _elevated(risk_map),
                "explanations": [e["chain_text"] for e in build_explanations(risk_map)],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed += 1

    # Flush any still-active contacts, then take the authoritative final snapshot.
    tail_timeline_before = len(pipeline.timeline)
    pipeline.finish()
    contacts_seen.extend(e for e in pipeline.timeline[tail_timeline_before:] if e["type"] == "contact")

    snapshot = pipeline.snapshot()
    snapshot["scenario"] = scenario
    snapshot["frames_processed"] = processed
    with open(snapshot_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2)

    checks = _evaluate_success(scenario, contacts_seen, snapshot)
    _write_summary(summary_path, scenario, processed, contacts_seen, snapshot, checks,
                   model_path=model_path)

    return {
        "scenario": scenario,
        "frames_processed": processed,
        "events_path": events_path,
        "snapshot_path": snapshot_path,
        "summary_path": summary_path,
        "contacts": contacts_seen,
        "snapshot": snapshot,
        "checks": checks,
        "physical_verification": snapshot.get("physical_verification", False),
    }


def _evaluate_success(scenario, contacts, snapshot) -> dict:
    """Check the flagship success criteria WITHOUT asserting exact probabilities
    (tendencies + chain correctness only). Non-flagship scenarios only report
    their contact/objects; the flagship-specific checks are marked n/a."""
    by_class = {o["class_name"]: o for o in snapshot["objects"]}
    seen_pairs = {_unordered_pair(c["source_class"], c["target_class"]) for c in contacts}

    checks = {"scenario": scenario, "is_flagship": scenario == "flagship_chain"}
    if scenario != "flagship_chain":
        checks["note"] = "flagship-specific checks skipped for non-flagship scenario"
        return checks

    expected_pairs = [_unordered_pair(a, b) for a, b in FLAGSHIP_PAIRS]
    checks["contact_pairs_present"] = {
        f"{a}<->{b}": (_unordered_pair(a, b) in seen_pairs) for a, b in FLAGSHIP_PAIRS
    }
    checks["all_three_contacts"] = all(p in seen_pairs for p in expected_pairs)
    checks["exactly_three_meaningful_contacts"] = len(contacts) == 3

    plate = by_class.get("plate")
    if plate is not None:
        risk_map = {o["track_id"]: o for o in snapshot["objects"]}
        chain_classes = [risk_map[t]["class_name"] for t in plate["risk_chain"] if t in risk_map]
        checks["plate_has_downstream_risk"] = plate["risk_score"] > 0.0
        checks["plate_chain"] = chain_classes
        checks["plate_chain_correct"] = (
            len(chain_classes) >= 2
            and chain_classes[0] == "nut_butter_jar"
            and chain_classes[-1] == "plate"
        )
        checks["plate_never_direct_source_contact"] = _unordered_pair("nut_butter_jar", "plate") not in seen_pairs
    else:
        checks["plate_has_downstream_risk"] = False
        checks["plate_chain_correct"] = False

    # Risk should rise along the chain then decay, staying > 0 downstream.
    checks["risk_tendency"] = {
        name: {"risk_class": by_class[name]["risk_class"], "risk_score": by_class[name]["risk_score"]}
        for name in ("nut_butter_jar", "cutlery", "bread", "plate") if name in by_class
    }
    checks["passed"] = bool(
        checks.get("all_three_contacts")
        and checks.get("plate_has_downstream_risk")
        and checks.get("plate_chain_correct")
        and checks.get("plate_never_direct_source_contact", True)
    )
    return checks


def _write_summary(path, scenario, processed, contacts, snapshot, checks, *, model_path):
    lines = []
    lines.append("TrackSense headless demo run (no camera, replay-based)")
    lines.append("=" * 56)
    lines.append(f"scenario            : {scenario}")
    lines.append(f"frames processed    : {processed}")
    lines.append(f"detection source    : {snapshot.get('source', snapshot.get('source_kind'))}")
    lines.append(f"detector model      : {snapshot.get('model')}")
    lines.append(f"rf model path       : {model_path or 'default (model/risk_random_forest.joblib)'}")
    lines.append(f"physical_verification: {snapshot.get('physical_verification', False)}")
    lines.append("")
    lines.append(f"Contact events ({len(contacts)}):")
    for c in contacts:
        lines.append(f"  {c['source_class']} <-> {c['target_class']}  "
                     f"dur={c['duration']}s overlap={c['overlap_ratio']} "
                     f"-> {c['risk_class']} ({c['risk_score']})")
    lines.append("")
    lines.append("Final object risk (risk desc):")
    for o in snapshot["objects"]:
        chain = " -> ".join(
            {ob["track_id"]: ob["class_name"] for ob in snapshot["objects"]}.get(t, "?")
            for t in o["risk_chain"]
        )
        lines.append(f"  {o['class_name']:<16}{o['risk_class']:<7}score={o['risk_score']:.3f} "
                     f"depth={o['propagation_depth']}  chain=[{chain}]")
    lines.append("")
    lines.append("Explanations (why an object is elevated downstream risk):")
    for e in snapshot["explanations"]:
        lines.append(f"  {e['object']}: {e['chain_text']}  ({e['risk_class']} {e['risk_score']})")
    if not snapshot["explanations"]:
        lines.append("  (none)")
    lines.append("")
    lines.append("Alerts:")
    for a in snapshot["alerts"]:
        lines.append(f"  {a['object']}: {a['chain_text']}  ({a['risk_class']} {a['risk_score']})")
    if not snapshot["alerts"]:
        lines.append("  (none)")
    lines.append("")
    lines.append("Success criteria:")
    if checks.get("is_flagship"):
        lines.append(f"  all three flagship contacts   : {checks.get('all_three_contacts')}")
        lines.append(f"  exactly three contacts        : {checks.get('exactly_three_meaningful_contacts')}")
        lines.append(f"  plate has downstream risk     : {checks.get('plate_has_downstream_risk')}")
        lines.append(f"  plate chain correct           : {checks.get('plate_chain_correct')} "
                     f"{checks.get('plate_chain')}")
        lines.append(f"  plate no direct source contact: {checks.get('plate_never_direct_source_contact')}")
        lines.append(f"  OVERALL PASSED                : {checks.get('passed')}")
    else:
        lines.append(f"  {checks.get('note')}")
    lines.append("")
    lines.append("NOTE: Physical kitchen/camera validation has not yet been completed. "
                 "These results confirm software integration and replay-based behavior, "
                 "not real-world contact accuracy.")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _console_report(result) -> None:
    """Print an ASCII-safe summary (Windows consoles choke on the arrow glyph)."""
    checks = result["checks"]
    print(f"scenario={result['scenario']} frames={result['frames_processed']} "
          f"physical_verification={result['physical_verification']}")
    print(f"contact events: {len(result['contacts'])}")
    for c in result["contacts"]:
        print(f"  {c['source_class']} <-> {c['target_class']}  "
              f"-> {c['risk_class']} ({c['risk_score']})")
    print("final object risk:")
    for o in result["snapshot"]["objects"]:
        print(f"  {o['class_name']:<16}{o['risk_class']:<7}score={o['risk_score']:.3f} "
              f"depth={o['propagation_depth']}")
    if checks.get("is_flagship"):
        chain = " -> ".join(checks.get("plate_chain", []) or [])
        print(f"plate chain: {chain}")
        print(f"FLAGSHIP SUCCESS: {checks.get('passed')}")
    print(f"artifacts:\n  {result['events_path']}\n  {result['snapshot_path']}\n  {result['summary_path']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Headless (no-camera) TrackSense demo runner.")
    parser.add_argument("--scenario", default="flagship_chain", choices=list_scenarios())
    parser.add_argument("--frames", type=int, default=None,
                        help="Cap frames processed (default: run the whole scenario).")
    parser.add_argument("--model", default=None, help="RF model path (default: configured RF model).")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    result = run(args.scenario, frames=args.frames, model_path=args.model,
                 fps=args.fps, seed=args.seed, out_dir=args.out_dir)
    _console_report(result)
    if result["checks"].get("is_flagship") and not result["checks"].get("passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
