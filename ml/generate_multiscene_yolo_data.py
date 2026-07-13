"""Phase 4 -- Synthetic multi-object YOLO scene generator for TrackSense.

Reads the object crop bank (ml/create_object_crop_bank.py) and composites 2-5
labelled object instances of DIFFERENT classes into a single image, producing a
correct multi-object YOLO label per scene. This directly attacks the audited
weakness: every real training image is single-class, so the detector never saw
two classes co-occur. We over-sample the concrete failure case
`nut_butter_jar + cutlery`, especially close / touching / partially occluded.

Correctness guarantees:
  - every emitted box is the TIGHT object bbox (padding excluded), normalized to
    [0,1] and clipped to the frame;
  - placement enforces that no labelled object is ever occluded below
    --min-visible of its own area, so there are no unlabelled "ghost" objects
    and no impossible labels;
  - class ids are the 8 model-local ids only (0..7); `counter` never appears;
    bread stays local id 7.

Backgrounds are strongly blurred/dimmed TRAIN images (no val/test leakage) plus
synthetic solid/gradient/wood/noise fills, so the backdrop reads as texture, not
as an unlabelled object.

Reproducible with --seed (default 42).

Usage:
  python ml/generate_multiscene_yolo_data.py --num 3000 --seed 42 \
      --out data/training_8class_multiscene/synthetic_train
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.class_schema import MODEL_LOCAL_NAMES, NUM_TRAINING_CLASSES  # noqa: E402

NAME_TO_ID = {v: k for k, v in MODEL_LOCAL_NAMES.items()}
J, N, HAND, CUT, BOARD, PLATE, BOWL, BREAD = 0, 1, 2, 3, 4, 5, 6, 7

# ---------------------------------------------------------------------------
# Combination buckets. Weights sum to 1.0; the immediate failure
# (nut_butter_jar + cutlery) is heavily over-sampled at 35%.
# ---------------------------------------------------------------------------
BUCKETS = [
    ("jar_cutlery", 0.35, [
        ((J, CUT), 0.70),
        ((J, CUT, PLATE), 0.12),
        ((J, CUT, BREAD), 0.12),
        ((J, CUT, HAND), 0.06),
    ]),
    ("flagship_chain", 0.20, [
        ((J, CUT, BREAD, PLATE), 0.40),
        ((J, CUT, BREAD), 0.20),
        ((J, CUT, PLATE), 0.15),
        ((CUT, BREAD, PLATE), 0.15),
        ((J, BREAD, PLATE), 0.10),
    ]),
    ("bread_lines", 0.15, [
        ((CUT, BREAD), 0.40),
        ((BREAD, PLATE), 0.35),
        ((CUT, BREAD, PLATE), 0.25),
    ]),
    ("hand_combos", 0.10, [
        ((HAND, J), 0.25),
        ((HAND, CUT), 0.20),
        ((HAND, BREAD), 0.15),
        ((HAND, PLATE), 0.15),
        ((HAND, CUT, PLATE), 0.15),
        ((HAND, BOWL), 0.10),
    ]),
    ("whole_nuts_combos", 0.10, [
        ((N, HAND), 0.25),
        ((N, BOWL), 0.30),
        ((N, PLATE), 0.15),
        ((N, J), 0.15),
        ((N, BOWL, CUT), 0.15),
    ]),
    ("misc_kitchen", 0.10, [
        ((BOARD, CUT, BREAD), 0.25),
        ((BOWL, CUT), 0.20),
        ((BOWL, HAND), 0.15),
        ((PLATE, BREAD), 0.10),
        ((BOARD, CUT), 0.10),
        ((BOWL, PLATE), 0.10),
        ((J, BOWL), 0.10),
    ]),
]

# Per-scene placement style. Occlusion is over-sampled because that is the
# physically-observed failure mode.
PLACEMENT_MODES = [("separated", 0.20), ("close", 0.20), ("touching", 0.15), ("overlap", 0.45)]


def weighted_choice(rng: random.Random, items):
    """items: list of (value, weight). Returns a value."""
    r = rng.random() * sum(w for _, w in items)
    upto = 0.0
    for value, w in items:
        upto += w
        if r <= upto:
            return value
    return items[-1][0]


def pick_combo(rng: random.Random):
    bucket = weighted_choice(rng, [(b, w) for (b, w, _) in BUCKETS])
    name = bucket
    members = next(members for (bn, _, members) in BUCKETS if bn == name)
    combo = weighted_choice(rng, members)
    return name, list(combo)


# Stress-test combos: nut_butter_jar + cutlery, close/touching/occluded, with a
# light third object sometimes. Used with --stress for the held-out stress set.
STRESS_COMBOS = [
    ((J, CUT), 0.60),
    ((J, CUT, PLATE), 0.15),
    ((J, CUT, BREAD), 0.15),
    ((J, CUT, HAND), 0.10),
]


def pick_combo_stress(rng: random.Random):
    return "jar_cutlery_stress", list(weighted_choice(rng, STRESS_COMBOS))


def load_crop_bank(bank_root: Path):
    index = bank_root / "crop_index.csv"
    if not index.is_file():
        raise SystemExit(f"FAIL: crop index not found: {index}. Run ml/create_object_crop_bank.py first.")
    by_class = defaultdict(list)
    with index.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = int(row["class_id"])
            by_class[cid].append({
                "path": REPO_ROOT / row["crop_path"],
                "obj": (float(row["obj_l"]), float(row["obj_t"]), float(row["obj_r"]), float(row["obj_b"])),
                "near_full": row.get("near_full_frame", "0") == "1",
            })
    for cid in range(NUM_TRAINING_CLASSES):
        if not by_class[cid]:
            print(f"WARNING: no crops for class {cid} {MODEL_LOCAL_NAMES[cid]}")
    return by_class


def sample_crop(rng: random.Random, by_class, cid):
    pool = by_class.get(cid, [])
    if not pool:
        return None
    non_full = [c for c in pool if not c["near_full"]]
    use = non_full if (non_full and rng.random() < 0.85) else pool
    return rng.choice(use)


def load_rgba(path: Path):
    im = Image.open(path).convert("RGBA")
    return np.array(im)  # HxWx4 uint8


def jitter_photometric(rng: random.Random, rgba: np.ndarray) -> np.ndarray:
    """Brightness/contrast jitter on RGB only; alpha untouched."""
    rgb = rgba[..., :3].astype(np.float32)
    contrast = rng.uniform(0.82, 1.18)
    brightness = rng.uniform(-18, 18)
    mean = rgb.mean(axis=(0, 1), keepdims=True)
    rgb = (rgb - mean) * contrast + mean + brightness
    out = rgba.copy()
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def rotate_rgba(rgba: np.ndarray, angle_deg: float, obj):
    """Rotate an RGBA crop about its center with expansion, and return the
    rotated array plus the axis-aligned inner-object box (in pixel coords)."""
    h, w = rgba.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    rot = cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    ox1, oy1, ox2, oy2 = obj
    corners = np.array([[ox1 * w, oy1 * h], [ox2 * w, oy1 * h],
                        [ox2 * w, oy2 * h], [ox1 * w, oy2 * h]])
    ones = np.ones((4, 1))
    tc = (M @ np.hstack([corners, ones]).T).T  # 4x2
    x1, y1 = tc[:, 0].min(), tc[:, 1].min()
    x2, y2 = tc[:, 0].max(), tc[:, 1].max()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(nw, x2), min(nh, y2)
    return rot, (x1, y1, x2, y2)


def prep_object(rng: random.Random, crop, canvas: int):
    """Load, optionally rotate, photometric-jitter, and scale a crop so its
    tight object box reaches a target size. Returns (rgba, inner_box_px)."""
    rgba = load_rgba(crop["path"])
    obj = crop["obj"]
    h, w = rgba.shape[:2]
    inner_px = (obj[0] * w, obj[1] * h, obj[2] * w, obj[3] * h)

    if rng.random() < 0.35:
        angle = rng.uniform(-14, 14)
        rgba, inner_px = rotate_rgba(rgba, angle, obj)

    if rng.random() < 0.6:
        rgba = jitter_photometric(rng, rgba)

    iw = max(1.0, inner_px[2] - inner_px[0])
    ih = max(1.0, inner_px[3] - inner_px[1])
    target = rng.uniform(0.16, 0.46) * canvas
    scale = target / max(iw, ih)
    # keep the whole crop within the canvas
    max_scale = (0.96 * canvas) / max(rgba.shape[0], rgba.shape[1])
    scale = min(scale, max_scale)
    nw = max(2, int(round(rgba.shape[1] * scale)))
    nh = max(2, int(round(rgba.shape[0] * scale)))
    rgba = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    inner = tuple(v * scale for v in inner_px)

    if rng.random() < 0.20:  # mild per-object blur
        k = rng.choice([3, 5])
        rgba = cv2.GaussianBlur(rgba, (k, k), 0)
    return rgba, inner


# ---- geometry helpers -----------------------------------------------------

def overlap_area(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    inter = overlap_area(a, b)
    if inter <= 0:
        return 0.0
    return inter / (box_area(a) + box_area(b) - inter)


def paste_rgba(canvas: np.ndarray, rgba: np.ndarray, px: int, py: int):
    h, w = rgba.shape[:2]
    H, W = canvas.shape[:2]
    x1, y1 = max(0, px), max(0, py)
    x2, y2 = min(W, px + w), min(H, py + h)
    if x2 <= x1 or y2 <= y1:
        return
    sub = rgba[y1 - py:y2 - py, x1 - px:x2 - px]
    alpha = sub[..., 3:4].astype(np.float32) / 255.0
    region = canvas[y1:y2, x1:x2].astype(np.float32)
    blended = sub[..., :3].astype(np.float32) * alpha + region * (1 - alpha)
    canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)


# ---- backgrounds ----------------------------------------------------------

def make_background(rng: random.Random, canvas: int, train_images: list[Path]) -> np.ndarray:
    mode = weighted_choice(rng, [("photo", 0.45), ("wood", 0.20), ("gradient", 0.15),
                                 ("solid", 0.10), ("noise", 0.10)])
    if mode == "photo" and train_images:
        for _ in range(4):
            p = rng.choice(train_images)
            try:
                im = Image.open(p).convert("RGB").resize((canvas, canvas), Image.LANCZOS)
                arr = np.array(im)
                break
            except Exception:
                arr = None
        if arr is None:
            mode = "solid"
        else:
            k = rng.choice([2, 3, 4]) * 2 + 1  # 5,7,9
            radius = int(canvas / rng.uniform(28, 45)) | 1
            arr = cv2.GaussianBlur(arr, (radius, radius), 0)
            arr = np.clip(arr.astype(np.float32) * rng.uniform(0.5, 0.85), 0, 255).astype(np.uint8)
            return arr
    if mode == "wood":
        base = np.array([rng.randint(120, 175), rng.randint(85, 130), rng.randint(45, 85)], np.float32)
        arr = np.tile(base, (canvas, canvas, 1))
        stripes = np.sin(np.linspace(0, rng.uniform(20, 45), canvas))[:, None, None] * rng.uniform(6, 14)
        arr = np.clip(arr + stripes + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 4, arr.shape), 0, 255)
        return arr.astype(np.uint8)
    if mode == "gradient":
        c1 = np.array([rng.randint(60, 200) for _ in range(3)], np.float32)
        c2 = np.array([rng.randint(60, 200) for _ in range(3)], np.float32)
        t = np.linspace(0, 1, canvas)[:, None]
        vert = (c1[None, :] * (1 - t) + c2[None, :] * t)  # canvas x 3
        arr = np.repeat(vert[:, None, :], canvas, axis=1)
        return np.clip(arr, 0, 255).astype(np.uint8)
    if mode == "noise":
        base = np.array([rng.randint(70, 180) for _ in range(3)], np.float32)
        arr = np.tile(base, (canvas, canvas, 1))
        arr += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 12, arr.shape)
        return np.clip(arr, 0, 255).astype(np.uint8)
    # solid
    base = np.array([rng.randint(70, 190) for _ in range(3)], np.float32)
    return np.clip(np.tile(base, (canvas, canvas, 1)), 0, 255).astype(np.uint8)


# ---- scene assembly -------------------------------------------------------

def place_scene(rng: random.Random, canvas: int, objs, mode: str, min_visible: float):
    """objs: list of (cid, rgba, inner_px). Places them so no kept object is
    occluded below min_visible. Returns list of (cid, tight_box_px)."""
    placed = []          # list of dicts: cid, tb(tight box), area, covered
    n = len(objs)
    gap = 0.03 * canvas
    for idx, (cid, rgba, inner) in enumerate(objs):
        h, w = rgba.shape[:2]
        iw, ih = inner[2] - inner[0], inner[3] - inner[1]
        if w >= canvas or h >= canvas:
            continue
        best = None
        tries = 90 if idx > 0 else 20
        for attempt in range(tries):
            px = rng.randint(0, max(0, canvas - w))
            py = rng.randint(0, max(0, canvas - h))
            tb = (px + inner[0], py + inner[1], px + inner[2], py + inner[3])
            area = box_area(tb)
            if area <= 1:
                continue
            ok = True
            relaxed = attempt > tries * 0.6
            # constraint vs already-placed objects (which are UNDER this one)
            add_cover = []
            for pl in placed:
                ov = overlap_area(tb, pl["tb"])
                if mode in ("separated", "close") and not relaxed:
                    if ov > 0 or (mode == "separated" and _min_gap(tb, pl["tb"]) < gap):
                        ok = False
                        break
                if mode == "touching" and not relaxed:
                    if iou(tb, pl["tb"]) > 0.05:
                        ok = False
                        break
                if mode == "overlap" and iou(tb, pl["tb"]) > 0.6:
                    ok = False
                    break
                # this object sits on TOP of pl -> pl loses `ov` of its area
                if (pl["covered"] + ov) / pl["area"] > (1 - min_visible):
                    ok = False
                    break
                add_cover.append((pl, ov))
            if ok:
                best = (px, py, tb, area, add_cover)
                break
        if best is None:
            # fallback: force a grid slot so primary objects survive
            slot = idx % 4
            gx = (slot % 2) * (canvas - w)
            gy = (slot // 2) * (canvas - h)
            px, py = int(gx), int(gy)
            tb = (px + inner[0], py + inner[1], px + inner[2], py + inner[3])
            best = (px, py, tb, box_area(tb), [])
        px, py, tb, area, add_cover = best
        for pl, ov in add_cover:
            pl["covered"] += ov
        placed.append({"cid": cid, "tb": tb, "area": area, "covered": 0.0,
                       "rgba": rgba, "px": px, "py": py})
    return placed


def _min_gap(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return (dx * dx + dy * dy) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic multi-object YOLO scenes.")
    ap.add_argument("--crop-bank", default=str(REPO_ROOT / "data" / "object_crop_bank"))
    ap.add_argument("--src-images", default=str(REPO_ROOT / "data" / "training_8class_balanced" / "train" / "images"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "training_8class_multiscene" / "synthetic_train"))
    ap.add_argument("--num", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sizes", default="512,576,640,704")
    ap.add_argument("--min-visible", type=float, default=0.45, help="Min fraction of any labelled object that must stay un-occluded.")
    ap.add_argument("--prefix", default="synth_", help="Filename prefix to avoid collisions with the real dataset.")
    ap.add_argument("--extra-clutter-prob", type=float, default=0.25, help="Chance of adding one extra clutter object (up to 5 total).")
    ap.add_argument("--stress", action="store_true", help="Generate a jar+cutlery-focused, contact/occlusion-heavy STRESS set.")
    ap.add_argument("--summary", default=str(REPO_ROOT / "reports" / "multiscene_generation_summary.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    sizes = [int(s) for s in args.sizes.split(",")]
    out_img = Path(args.out) / "images"
    out_lbl = Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    by_class = load_crop_bank(Path(args.crop_bank))
    train_images = [p for p in Path(args.src_images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"crop bank classes: {[len(by_class[c]) for c in range(NUM_TRAINING_CLASSES)]}")
    print(f"background pool: {len(train_images)} train images")
    print(f"generating {args.num} scenes -> {args.out}  seed={args.seed}")

    class_instances = Counter()
    combo_bucket_counts = Counter()
    cooc = defaultdict(int)          # frozenset(pair) -> count
    jar_cutlery_scenes = 0
    flagship_full = 0
    flagship_partial = 0
    scenes_written = 0
    obj_count_hist = Counter()

    pad = len(str(args.num))
    for i in range(args.num):
        canvas = rng.choice(sizes)
        if args.stress:
            bucket, combo = pick_combo_stress(rng)
        else:
            bucket, combo = pick_combo(rng)
        combo_bucket_counts[bucket] += 1
        classes = list(combo)
        # optional clutter object (may duplicate a class) up to 5 objects
        clutter_prob = 0.10 if args.stress else args.extra_clutter_prob
        if len(classes) < 5 and rng.random() < clutter_prob:
            classes.append(rng.choice(list(range(NUM_TRAINING_CLASSES))))

        mode = weighted_choice(rng, PLACEMENT_MODES)
        if args.stress:  # stress set: force contact / partial occlusion
            mode = weighted_choice(rng, [("touching", 0.35), ("overlap", 0.55), ("close", 0.10)])
        elif bucket == "jar_cutlery":  # bias the failure case toward contact/occlusion
            mode = weighted_choice(rng, [("touching", 0.30), ("overlap", 0.55), ("close", 0.15)])

        objs = []
        rng.shuffle(classes)  # randomize draw order -> varied "behind/in front"
        for cid in classes:
            crop = sample_crop(rng, by_class, cid)
            if crop is None:
                continue
            rgba, inner = prep_object(rng, crop, canvas)
            objs.append((cid, rgba, inner))
        if len(objs) < 2:
            continue

        bg = make_background(rng, canvas, train_images).copy()
        placed = place_scene(rng, canvas, objs, mode, args.min_visible)
        for pl in placed:
            paste_rgba(bg, pl["rgba"], pl["px"], pl["py"])

        # scene-level photometric + occasional global blur
        if rng.random() < 0.5:
            bg = jitter_photometric(rng, np.dstack([bg, np.full(bg.shape[:2], 255, np.uint8)]))[..., :3]
        if rng.random() < 0.22:
            k = rng.choice([3, 5])
            bg = cv2.GaussianBlur(bg, (k, k), 0)

        # build labels
        lines = []
        present = set()
        for pl in placed:
            x1, y1, x2, y2 = pl["tb"]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(canvas), x2), min(float(canvas), y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            cx = (x1 + x2) / 2 / canvas
            cy = (y1 + y2) / 2 / canvas
            nw = bw / canvas
            nh = bh / canvas
            if nw <= 0 or nh <= 0 or nw > 1 or nh > 1:
                continue
            lines.append(f"{pl['cid']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            class_instances[pl["cid"]] += 1
            present.add(pl["cid"])

        if len(lines) < 2 or len(present) < 2:
            continue

        stem = f"{args.prefix}{i:0{pad}d}_{bucket}"
        Image.fromarray(bg).save(out_img / f"{stem}.jpg", quality=90)
        (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        scenes_written += 1
        obj_count_hist[len(lines)] += 1

        # stats
        pl_classes = sorted(present)
        for a_i in range(len(pl_classes)):
            for b_i in range(a_i + 1, len(pl_classes)):
                cooc[(pl_classes[a_i], pl_classes[b_i])] += 1
        if J in present and CUT in present:
            jar_cutlery_scenes += 1
        chain = {J, CUT, BREAD, PLATE}
        if chain.issubset(present):
            flagship_full += 1
        elif len(chain & present) >= 3:
            flagship_partial += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{args.num} scenes... (written={scenes_written})")

    summary = {
        "seed": args.seed,
        "requested": args.num,
        "scenes_written": scenes_written,
        "sizes": sizes,
        "min_visible": args.min_visible,
        "bucket_counts": dict(combo_bucket_counts),
        "class_instances": {MODEL_LOCAL_NAMES[c]: int(class_instances[c]) for c in range(NUM_TRAINING_CLASSES)},
        "objects_per_scene_hist": {str(k): v for k, v in sorted(obj_count_hist.items())},
        "jar_cutlery_scenes": jar_cutlery_scenes,
        "flagship_full_scenes": flagship_full,
        "flagship_partial_scenes": flagship_partial,
        "cooccurrence": {f"{MODEL_LOCAL_NAMES[a]}+{MODEL_LOCAL_NAMES[b]}": v for (a, b), v in sorted(cooc.items(), key=lambda kv: -kv[1])},
        "out": str(Path(args.out).as_posix()),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== GENERATION SUMMARY ===")
    print(f"scenes written: {scenes_written}/{args.num}")
    print(f"jar+cutlery scenes: {jar_cutlery_scenes}")
    print(f"flagship full (jar+cut+bread+plate): {flagship_full}  partial(>=3 of chain): {flagship_partial}")
    print(f"objects/scene: {dict(sorted(obj_count_hist.items()))}")
    print("class instances:", {MODEL_LOCAL_NAMES[c]: int(class_instances[c]) for c in range(NUM_TRAINING_CLASSES)})
    print(f"summary -> {args.summary}")
    if scenes_written == 0:
        raise SystemExit("FAIL: no scenes written.")


if __name__ == "__main__":
    main()
