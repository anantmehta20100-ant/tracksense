"""Phase 11 -- Multi-object co-detection evaluation for the retrained model.

The headline metric is NOT mAP; it is: does the detector find nut_butter_jar AND
cutlery IN THE SAME FRAME? This script runs a checkpoint over

  - the held-out synthetic stress set (data/training_8class_multiscene/stress_test),
  - reports/yolo_smoke/peanut_cutlery_test.jpg (+ *_A / *_B variants if present),
  - any real photos in an --extra directory,

and reports, per model:
  - jar+cutlery CO-DETECTION rate,
  - average confidence for jar and for cutlery,
  - single-object failures (only one of the pair detected),
  - annotated outputs under reports/multiscene_detection_eval/<model_tag>/.

Pass --compare-model to A/B two checkpoints in one run (e.g. the retrained model
vs the single-object BACKUP) and print a side-by-side delta -- the intended way
to show the fix worked.

This script does NOT retrain or modify any weights. It refuses to run on a
checkpoint whose schema is not the TrackSense 8-class schema.

Usage (after Kaggle retraining):
  python vision/test_multiscene_detection.py \
      --model model/checkpoints/tracksense_8class_multiscene_best.pt \
      --compare-model model/checkpoints/tracksense_8class_singleobject_BACKUP.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vision.validate_yolo_checkpoint import check_schema  # noqa: E402

JAR, CUTLERY = 0, 3
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EVAL_ROOT = REPO_ROOT / "reports" / "multiscene_detection_eval"


def gather_images(stress_dir: Path, smoke_dir: Path, extra: Path | None):
    imgs = []
    sd = stress_dir / "images"
    if sd.is_dir():
        imgs += [("stress", p) for p in sorted(sd.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    for name in ("peanut_cutlery_test.jpg", "peanut_cutlery_A.jpg", "peanut_cutlery_B.jpg"):
        p = smoke_dir / name
        if p.is_file():
            imgs.append(("smoke", p))
    if extra and extra.is_dir():
        imgs += [("real", p) for p in sorted(extra.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return imgs


def eval_model(model_path: str, images, conf: float, tag: str):
    from ultralytics import YOLO
    model = YOLO(model_path)
    ok, names, checks = check_schema(model.names)
    if not ok:
        print(f"[{tag}] REFUSING: checkpoint schema is not the TrackSense 8-class schema.")
        for label, passed, detail in checks:
            if not passed:
                print(f"    [FAIL] {label} ({detail})")
        return None

    out_dir = EVAL_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    per_source = {}
    records = []
    for source, path in images:
        res = model(str(path), conf=conf, verbose=False)[0]
        det_conf = {JAR: [], CUTLERY: []}
        for box in res.boxes:
            cid = int(box.cls[0])
            if cid in det_conf:
                det_conf[cid].append(float(box.conf[0]))
        has_jar = len(det_conf[JAR]) > 0
        has_cut = len(det_conf[CUTLERY]) > 0
        rec = {
            "source": source, "image": path.name,
            "jar_detected": has_jar, "cutlery_detected": has_cut,
            "both_detected": has_jar and has_cut,
            "jar_conf": round(max(det_conf[JAR]), 3) if has_jar else 0.0,
            "cutlery_conf": round(max(det_conf[CUTLERY]), 3) if has_cut else 0.0,
        }
        records.append(rec)
        per_source.setdefault(source, []).append(rec)
        res.save(filename=str(out_dir / f"{path.stem}_det.jpg"))

    def summarize(recs):
        n = len(recs)
        if n == 0:
            return {}
        both = sum(r["both_detected"] for r in recs)
        jar_only = sum(r["jar_detected"] and not r["cutlery_detected"] for r in recs)
        cut_only = sum(r["cutlery_detected"] and not r["jar_detected"] for r in recs)
        neither = sum(not r["jar_detected"] and not r["cutlery_detected"] for r in recs)
        jar_confs = [r["jar_conf"] for r in recs if r["jar_detected"]]
        cut_confs = [r["cutlery_conf"] for r in recs if r["cutlery_detected"]]
        return {
            "n": n,
            "both_detected": both,
            "co_detection_rate": round(both / n, 3),
            "jar_only": jar_only, "cutlery_only": cut_only, "neither": neither,
            "avg_jar_conf": round(sum(jar_confs) / len(jar_confs), 3) if jar_confs else 0.0,
            "avg_cutlery_conf": round(sum(cut_confs) / len(cut_confs), 3) if cut_confs else 0.0,
        }

    summary = {
        "model": model_path, "tag": tag, "conf": conf,
        "overall": summarize(records),
        "by_source": {src: summarize(recs) for src, recs in per_source.items()},
        "annotated_dir": str(out_dir.as_posix()),
    }
    (out_dir / "summary.json").write_text(json.dumps({**summary, "records": records}, indent=2), encoding="utf-8")
    return summary


def print_summary(s):
    o = s["overall"]
    print(f"\n[{s['tag']}]  model={s['model']}")
    print(f"  overall co-detection: {o.get('both_detected',0)}/{o.get('n',0)} "
          f"({o.get('co_detection_rate',0):.3f})  "
          f"avg_jar_conf={o.get('avg_jar_conf',0):.3f}  avg_cutlery_conf={o.get('avg_cutlery_conf',0):.3f}")
    print(f"  jar_only={o.get('jar_only',0)}  cutlery_only={o.get('cutlery_only',0)}  neither={o.get('neither',0)}")
    for src, ss in s["by_source"].items():
        print(f"    {src:6}: co-det {ss.get('both_detected',0)}/{ss.get('n',0)} ({ss.get('co_detection_rate',0):.3f})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate jar+cutlery multi-object co-detection.")
    ap.add_argument("--model", default=str(REPO_ROOT / "model" / "checkpoints" / "tracksense_8class_multiscene_best.pt"),
                    help="Checkpoint to evaluate (default: the retrained multiscene model path).")
    ap.add_argument("--compare-model", default=None, help="Optional second checkpoint for A/B (e.g. the single-object BACKUP).")
    ap.add_argument("--stress-dir", default=str(REPO_ROOT / "data" / "training_8class_multiscene" / "stress_test"))
    ap.add_argument("--smoke-dir", default=str(REPO_ROOT / "reports" / "yolo_smoke"))
    ap.add_argument("--extra", default=None, help="Directory of real physical photos to also test.")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    images = gather_images(Path(args.stress_dir), Path(args.smoke_dir), Path(args.extra) if args.extra else None)
    if not images:
        print("No evaluation images found (stress set / smoke image / --extra).")
        return 2
    print(f"Evaluating on {len(images)} images "
          f"({sum(1 for s,_ in images if s=='stress')} stress, "
          f"{sum(1 for s,_ in images if s=='smoke')} smoke, "
          f"{sum(1 for s,_ in images if s=='real')} real).")

    summaries = {}
    for tag, path in [("model", args.model)] + ([("baseline", args.compare_model)] if args.compare_model else []):
        if not os.path.exists(path):
            print(f"[{tag}] checkpoint missing: {path}  (skipping)")
            continue
        s = eval_model(path, images, args.conf, tag)
        if s:
            summaries[tag] = s
            print_summary(s)

    if "model" in summaries and "baseline" in summaries:
        m = summaries["model"]["overall"]["co_detection_rate"]
        b = summaries["baseline"]["overall"]["co_detection_rate"]
        print(f"\n=== A/B DELTA ===\n  new model co-detection: {m:.3f}\n  baseline co-detection : {b:.3f}\n  improvement: {m-b:+.3f}")

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    (EVAL_ROOT / "eval_summary.json").write_text(
        json.dumps({tag: {k: v for k, v in s.items() if k != "records"} for tag, s in summaries.items()}, indent=2),
        encoding="utf-8")
    print(f"\nEval outputs -> {EVAL_ROOT}")
    return 0 if summaries else 2


if __name__ == "__main__":
    raise SystemExit(main())
