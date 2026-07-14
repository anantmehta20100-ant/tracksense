"""Live YOLO -> contact -> RF -> RiskEngine runner (real-camera analogue of
pipeline/headless_demo_runner.py).

Feeds REAL detections from the 8-class detector (tracksense_8class_best.pt)
through the SAME RiskPipeline the mock demo uses, so a physical contact between
two real objects produces a real ContactEvent and RF-scored risk -- nothing here
fabricates risk or bypasses the contact tracker / RF / RiskEngine:

    YoloDetectionSource.frames()   (real camera / video / image -> FrameData)
      -> RiskPipeline.process_frame()  (IoUTracker -> ContactTracker -> RF -> RiskEngine)
      -> per-frame log + final snapshot

The detector is validated on load (rejects the old single-class best.pt / any
non-8-class schema). Use it for the first physical verification of a contact link
(e.g. nut_butter_jar -> cutlery) without needing the whole flagship chain.

This runner records real camera input as first-link physical evidence but never
claims the FULL nut_butter_jar -> cutlery -> bread -> plate chain is physically
validated -- that needs bread and plate too.

CLI:
    python pipeline/live_yolo_runner.py --camera 0 --frames 60
    python pipeline/live_yolo_runner.py --video path/to/clip.mp4
    python pipeline/live_yolo_runner.py --image reports/yolo_smoke/peanut_cutlery_test.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

from config.runtime_config import CONTACT, RF_MODEL_PATH, YOLO_MODEL_PATH  # noqa: E402
from pipeline.propagation import build_explanations  # noqa: E402
from pipeline.risk_pipeline import RiskPipeline  # noqa: E402

DEFAULT_OUT_DIR = os.path.join("reports", "physical_peanut_cutlery")


def _elevated(risk_map) -> list:
    out = [o for o in risk_map.values() if o["risk_class"] != "LOW"]
    out.sort(key=lambda o: o["risk_score"], reverse=True)
    return [{"class_name": o["class_name"], "track_id": o["track_id"],
             "risk_class": o["risk_class"], "risk_score": o["risk_score"]} for o in out]


def _detections_from_result(result, names, frame_index, timestamp):
    """Build canonical Detection objects from one raw YOLO Results object -- mirrors
    YoloDetectionSource.detect so the preview path agrees with the headless path."""
    from config.allergens import OBJECT_CLASS_TO_ID
    from pipeline.contracts import Detection

    dets = []
    for box in result.boxes:
        local_id = int(box.cls[0])
        class_name = names.get(local_id, str(local_id))
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        dets.append(Detection(
            class_id=OBJECT_CLASS_TO_ID.get(class_name, -1), class_name=class_name,
            confidence=max(0.0, min(1.0, conf)), bbox_xyxy=(x1, y1, x2, y2),
            frame_index=frame_index, timestamp=timestamp))
    return dets


def inflate_result_boxes(result, margin=0.08):
    """Pad each detected box outward by `margin` of its own size (clamped to the
    frame) so the drawn boxes read a little larger. VISUAL ONLY -- call it AFTER
    detections have been extracted; it does not affect tracked/risk coordinates."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return result
    data = boxes.data.clone()  # YOLO returns inference tensors -> clone before in-place edits
    h, w = int(result.orig_shape[0]), int(result.orig_shape[1])
    for row in data:
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        pw = (x2 - x1) * margin
        ph = (y2 - y1) * margin
        row[0] = max(0.0, x1 - pw)
        row[1] = max(0.0, y1 - ph)
        row[2] = min(float(w - 1), x2 + pw)
        row[3] = min(float(h - 1), y2 + ph)
    boxes.data = data  # swap the enlarged boxes in so plot()/overlay use them
    return result


def draw_collision_overlay(img, result):
    """Highlight where two detected bounding boxes touch/overlap on the annotated
    frame -- a live visual cue that objects are in contact. Fills each overlapping
    region with a translucent red patch, outlines it, and tags it "CONTACT". This
    is pure per-frame geometry on the current boxes (it does NOT run the contact
    tracker), so it flags touching boxes the instant they overlap. Returns `img`."""
    import cv2

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) < 2:
        return img
    rects = [tuple(b.xyxy[0].tolist()) for b in boxes]
    hits = []
    for i in range(len(rects)):
        ax1, ay1, ax2, ay2 = rects[i]
        for j in range(i + 1, len(rects)):
            bx1, by1, bx2, by2 = rects[j]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 > ix1 and iy2 > iy1:      # boxes overlap -> intersection rectangle
                hits.append((int(ix1), int(iy1), int(ix2), int(iy2)))
    if not hits:
        return img
    overlay = img.copy()
    for ix1, iy1, ix2, iy2 in hits:
        cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)   # translucent red fill
    for ix1, iy1, ix2, iy2 in hits:
        cv2.rectangle(img, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)
        ty = max(14, iy1 - 6)
        cv2.putText(img, "CONTACT", (ix1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, "CONTACT", (ix1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return img


def _overlapping_pairs(result):
    """Parse one YOLO result into ([(classA, classB), ...], [(x1,y1,x2,y2,class), ...]):
    the class pairs whose boxes overlap this frame, plus per-box geometry+class.
    Same overlap test as draw_collision_overlay, but class-aware so the
    contamination tracker can reason about WHAT touched what."""
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {}) or {}
    meta = []
    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            meta.append((x1, y1, x2, y2, names.get(int(b.cls[0]), str(int(b.cls[0])))))
    pairs = []
    for i in range(len(meta)):
        ax1, ay1, ax2, ay2, ac = meta[i]
        for j in range(i + 1, len(meta)):
            bx1, by1, bx2, by2, bc = meta[j]
            if min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1):
                pairs.append((ac, bc))
    return pairs, meta


def _draw_contam_banner(img, infected, new_infections, new_allergens):
    """Bottom status line of infected items; a top flash bar on a NEW event --
    red for a new infection, amber for a first allergen detection."""
    import cv2

    h, w = img.shape[:2]
    flash = None
    if new_infections:
        flash = ("! CONTAMINATION: " + ", ".join(new_infections), (0, 0, 200))     # red (BGR)
    elif new_allergens:
        flash = ("! ALLERGEN DETECTED: " + ", ".join(new_allergens), (0, 130, 220))  # amber (BGR)
    if flash:
        msg, col = flash
        bar = img.copy()
        cv2.rectangle(bar, (0, 0), (w, 42), col, -1)
        cv2.addWeighted(bar, 0.85, img, 0.15, 0, img)
        cv2.putText(img, msg, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    line = ("INFECTED: " + ", ".join(infected)) if infected else "no contamination detected"
    cv2.putText(img, line, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, line, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if infected else (0, 190, 0), 2)


def draw_contamination_overlay(img, result, tracker, *, frame_index=0, timestamp=0.0):
    """Contamination-aware replacement for draw_collision_overlay. Updates
    `tracker` with this frame's touching class pairs, then annotates the frame:
    translucent red on each contact patch, a SOURCE/INFECTED tag + coloured box
    on every carrier, and a banner of infected items (flashing on a new one).
    Returns `img`. Pure per-frame geometry + the sticky class memory."""
    import cv2

    pairs, meta = _overlapping_pairs(result)
    boxes = {m[4]: (m[0], m[1], m[2], m[3]) for m in meta}   # class -> bbox, for risk grading
    newly = tracker.observe(pairs, frame_index=frame_index, timestamp=timestamp,
                            present_classes=[m[4] for m in meta], boxes=boxes)

    # Translucent red fill on the touching regions (contact cue).
    if pairs:
        overlay = img.copy()
        for i in range(len(meta)):
            ax1, ay1, ax2, ay2, _ = meta[i]
            for j in range(i + 1, len(meta)):
                bx1, by1, bx2, by2, _ = meta[j]
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Tag every carrier box by status (clean boxes keep YOLO's own label only).
    for x1, y1, x2, y2, cls in meta:
        status = tracker.status(cls)
        if status == "source":
            color, tag = (0, 140, 255), "PEANUT SOURCE"      # orange
        elif status == "infected":
            color, tag = (0, 0, 255), "INFECTED"             # red
        else:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = (f"{tag}: {cls} ({tracker.risk_of(cls):.2f})"
                 if status == "infected" else f"{tag}: {cls}")
        ty = max(16, y1 - 8)
        cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _draw_contam_banner(img, sorted(tracker.infected), newly,
                        getattr(tracker, "new_allergens", []))
    return img


def _draw_overlay(img, frame_classes, expect_pair, contact_active):
    """Draw a readable status HUD (black outline + colour) onto the annotated frame."""
    import cv2

    both = all(c in frame_classes for c in expect_pair)
    lines = [
        (f"detected: {', '.join(frame_classes) or 'none'}", (0, 255, 255)),
        (f"both {expect_pair[0]} + {expect_pair[1]}: {'YES' if both else 'no'}",
         (0, 210, 0) if both else (0, 0, 255)),
    ]
    if contact_active:
        lines.append((f"CONTACT! {expect_pair[0]} -> {expect_pair[1]} (cutlery elevated)", (0, 230, 0)))
    else:
        lines.append(("hold jar + cutlery TOUCHING and still", (230, 230, 230)))
    y = 28
    for text, color in lines:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        y += 30


def _preview_loop(source, *, camera, video, frames, record, expect_pair, imgsz=None):
    """Show a live annotated window and feed each frame through `record`. Stops on
    a detected first-link contact (after a brief success hold), 'q', or the cap.

    Latency control: keep the capture buffer at 1 frame (always grab the freshest,
    never a backlog behind slow CPU inference) and run detection at a smaller
    `imgsz` so the window stays responsive."""
    import cv2

    from vision.mock_detection_source import FrameData

    src = video if video is not None else (0 if camera is None else camera)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"PREVIEW: could not open source {src!r}.")
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # drop stale buffered frames -> less lag
    except Exception:
        pass
    infer_imgsz = int(imgsz) if imgsz else 448   # smaller = faster inference = smoother preview
    window = "TrackSense first-link  (press q to quit)"
    cap_frames = frames if frames and frames > 0 else 100000
    frame_index = 0
    success_hold = 0
    print(f"PREVIEW window open (imgsz={infer_imgsz}) -- position jar + cutlery TOUCHING; "
          "auto-stops on contact, or press q.")
    try:
        while frame_index < cap_frames:
            ok, frame = cap.read()
            if not ok:
                break
            result = source.model(frame, verbose=False, imgsz=infer_imgsz)[0]
            dets = _detections_from_result(result, source.names, frame_index, frame_index / 30.0)
            fd = FrameData(frame_index=frame_index, timestamp=frame_index / 30.0,
                           detections=dets, control_events=[])
            new_contacts, frame_classes = record(fd)
            # Only celebrate / auto-stop on the REAL target pair (ignore spurious
            # same-class contacts like cutlery<->cutlery from double detections).
            if any(frozenset((c["source_class"], c["target_class"])) == frozenset(expect_pair)
                   for c in new_contacts):
                success_hold = 25
            display = result.plot()
            _draw_overlay(display, frame_classes, expect_pair, success_hold > 0)
            cv2.imshow(window, display)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            if success_hold > 0:
                success_hold -= 1
                if success_hold == 0:
                    break
            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run(*, model_path: str = None, camera: int = None, video: str = None, image: str = None,
        frames: int = 60, rf_model_path=None, out_dir: str = DEFAULT_OUT_DIR,
        expect_pair=("nut_butter_jar", "cutlery"), show: bool = False,
        conf: float = None, persistence: int = None, imgsz: int = None) -> dict:
    """Drive the real YOLO detector through the risk pipeline. Returns run
    metadata + evidence. Raises FileNotFoundError / ModelSchemaMismatch on a
    missing or wrong checkpoint (fail-fast, never silent).

    conf/persistence are live-capture tolerances (not applied to the mock demo):
    `conf` lowers the detector's confidence floor so a flickering webcam co-detects
    both objects more often; `persistence` shortens the contact debounce for the
    higher, noisier webcam frame rate. Proximity of REAL detected boxes is still
    required -- these never fabricate a contact."""
    model_path = model_path or YOLO_MODEL_PATH
    rf_model_path = rf_model_path if rf_model_path is not None else RF_MODEL_PATH

    # Import lazily so this module stays importable without ultralytics/cv2.
    from vision.yolo_detection_source import YoloDetectionSource

    if image is not None:
        source = YoloDetectionSource(model_path=model_path, video_path=image)
        input_kind = f"image:{os.path.basename(image)}"
    elif video is not None:
        source = YoloDetectionSource(model_path=model_path, video_path=video)
        input_kind = f"video:{os.path.basename(video)}"
    else:
        cam = 0 if camera is None else camera
        source = YoloDetectionSource(model_path=model_path, camera_index=cam)
        input_kind = f"camera:{cam}"

    if conf is not None:
        source.model.overrides["conf"] = float(conf)   # lower detector floor for live capture

    contact_config = CONTACT if persistence is None else replace(
        CONTACT, start_persistence_frames=int(persistence),
        end_persistence_frames=min(CONTACT.end_persistence_frames, max(1, int(persistence))))
    pipeline = RiskPipeline(model_path=rf_model_path, contact_config=contact_config, source_kind="yolo")

    os.makedirs(out_dir, exist_ok=True)
    events_path = os.path.join(out_dir, "contact_events.jsonl")
    detect_path = os.path.join(out_dir, "camera_detection_result.json")
    snapshot_path = os.path.join(out_dir, "final_snapshot.json")

    contacts_seen = []
    detected_classes = set()
    counters = {"processed": 0, "frames_with_both": 0}
    per_frame = []
    handle = open(events_path, "w", encoding="utf-8")

    def record(fd):
        """Process one frame through the pipeline + append its evidence row.
        Returns (new_contacts, frame_classes) for the optional live overlay."""
        timeline_before = len(pipeline.timeline)
        pipeline.process_frame(fd)
        new_contacts = [e for e in pipeline.timeline[timeline_before:] if e["type"] == "contact"]
        contacts_seen.extend(new_contacts)
        frame_classes = sorted({t.class_name for t in pipeline.tracks})
        detected_classes.update(frame_classes)
        if all(c in frame_classes for c in expect_pair):
            counters["frames_with_both"] += 1
        row = {
            "frame_index": fd.frame_index,
            "detections": [
                {"class_name": d.class_name, "confidence": round(d.confidence, 4),
                 "bbox_xyxy": [round(v, 1) for v in d.bbox_xyxy]}
                for d in fd.detections
            ],
            "tracked_objects": [
                {"track_id": t.track_id, "class_name": t.class_name} for t in pipeline.tracks
            ],
            "contacts": [
                {"source_class": c["source_class"], "target_class": c["target_class"],
                 "risk_class": c["risk_class"], "risk_score": c["risk_score"]}
                for c in new_contacts
            ],
        }
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        per_frame.append(row)
        counters["processed"] += 1
        _print_frame(row)
        return new_contacts, frame_classes

    try:
        if show and image is None:
            _preview_loop(source, camera=camera, video=video, frames=frames,
                          record=record, expect_pair=expect_pair, imgsz=imgsz)
        else:
            for fd in source.frames():
                if counters["processed"] >= max(1, frames):
                    break
                record(fd)
    finally:
        handle.close()

    processed = counters["processed"]
    frames_with_both = counters["frames_with_both"]

    tail_before = len(pipeline.timeline)
    pipeline.finish()
    contacts_seen.extend(e for e in pipeline.timeline[tail_before:] if e["type"] == "contact")

    snapshot = pipeline.snapshot()
    snapshot["input"] = input_kind
    snapshot["frames_processed"] = processed
    with open(snapshot_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2)

    # Detection-only evidence file (which classes the real detector saw).
    with open(detect_path, "w", encoding="utf-8") as handle:
        json.dump({
            "input": input_kind,
            "model": os.path.basename(model_path),
            "frames_processed": processed,
            "detected_classes": sorted(detected_classes),
            "frames_with_both_expected": frames_with_both,
            "expected_pair": list(expect_pair),
        }, handle, ensure_ascii=False, indent=2)

    seen_pairs = {frozenset((c["source_class"], c["target_class"])) for c in contacts_seen}
    first_link_contact = frozenset(expect_pair) in seen_pairs
    risk_map = {o["track_id"]: o for o in snapshot["objects"]}
    by_class = {o["class_name"]: o for o in snapshot["objects"]}
    target = expect_pair[1]
    target_state = by_class.get(target)
    target_elevated = bool(target_state and target_state["risk_class"] != "LOW")
    target_chain = ([risk_map[t]["class_name"] for t in target_state["risk_chain"]]
                    if target_state else [])

    result = {
        "input": input_kind,
        "frames_processed": processed,
        "detected_classes": sorted(detected_classes),
        "both_expected_detected": all(c in detected_classes for c in expect_pair),
        "frames_with_both_expected": frames_with_both,
        "contacts": contacts_seen,
        "first_link_contact_detected": first_link_contact,
        "target_elevated": target_elevated,
        "target_chain": target_chain,
        "elevated_objects": _elevated({o["track_id"]: o for o in snapshot["objects"]}),
        "explanations": [e["chain_text"] for e in build_explanations(
            {o["track_id"]: o for o in snapshot["objects"]})],
        "snapshot": snapshot,
        "events_path": events_path,
        "snapshot_path": snapshot_path,
        "detect_path": detect_path,
        # Honest scoping: this run can prove the FIRST link physically, never the
        # full chain (needs bread + plate).
        "first_link_physically_verified": bool(first_link_contact and target_elevated),
        "full_chain_physical_verification": False,
    }
    _print_result(result, expect_pair)
    return result


def _print_frame(row) -> None:
    dets = ", ".join(f"{d['class_name']}({d['confidence']:.2f})" for d in row["detections"]) or "(none)"
    line = f"f{row['frame_index']:>3}  {dets}"
    for c in row["contacts"]:
        line += f"  | CONTACT {c['source_class']}<->{c['target_class']} -> {c['risk_class']} ({c['risk_score']})"
    print(line)


def _print_result(result, expect_pair) -> None:
    a, b = expect_pair
    print("-" * 60)
    print(f"input                        : {result['input']}")
    print(f"frames processed             : {result['frames_processed']}")
    print(f"detected classes             : {result['detected_classes']}")
    print(f"both {a}+{b} detected         : {result['both_expected_detected']}")
    print(f"frames with both in view     : {result['frames_with_both_expected']}")
    print(f"first-link contact detected  : {result['first_link_contact_detected']}")
    print(f"{b} elevated                 : {result['target_elevated']}")
    print(f"{b} chain                    : {' -> '.join(result['target_chain'])}")
    ascii_expl = [e.replace("→", "->") for e in result["explanations"]]
    print(f"explanations                 : {ascii_expl}")
    print(f"FIRST-LINK PHYSICALLY VERIFIED: {result['first_link_physically_verified']}")
    print(f"full-chain physical verified : {result['full_chain_physical_verification']} (needs bread + plate)")
    print(f"artifacts:\n  {result['events_path']}\n  {result['snapshot_path']}\n  {result['detect_path']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Live YOLO -> contact -> RF risk runner.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None, help="Camera index (default 0).")
    src.add_argument("--video", default=None, help="Path to a video file.")
    src.add_argument("--image", default=None, help="Path to a single image (detection only).")
    parser.add_argument("--model", default=None, help="YOLO weights (default: configured 8-class model).")
    parser.add_argument("--rf-model", default=None, help="RF model path (default: configured RF model).")
    parser.add_argument("--frames", type=int, default=60, help="Max frames to process (camera/video).")
    parser.add_argument("--show", action="store_true",
                        help="Open a live preview window with detection boxes + status (camera/video).")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detector confidence floor for live capture (e.g. 0.15). Lower = both "
                             "objects co-detected more often on a flickering webcam.")
    parser.add_argument("--persistence", type=int, default=None,
                        help="Consecutive close frames needed to confirm a contact (default 5). "
                             "Lower (e.g. 3) suits the noisier live webcam frame rate.")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Inference image size for the preview (default 448). Smaller = "
                             "faster/smoother, e.g. 320.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    result = run(model_path=args.model, camera=args.camera, video=args.video, image=args.image,
                 frames=args.frames, rf_model_path=args.rf_model, out_dir=args.out_dir, show=args.show,
                 conf=args.conf, persistence=args.persistence, imgsz=args.imgsz)
    return 0 if result["first_link_contact_detected"] or args.image is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
