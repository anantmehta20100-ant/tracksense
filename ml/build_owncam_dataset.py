"""Add hand-checked frames from the real camera to the 8-class training set.

    python ml/build_owncam_dataset.py \
        --base data/training_8class_multiscene \
        --train-sessions data/own_frames/s1 data/own_frames/s2 \
        --test-sessions  data/own_frames/test1

Each session folder is what ml/prelabel_frames.py leaves behind, after the
boxes were corrected: <session>/images/*.jpg + <session>/labels/*.txt in the
8-class model-local ids.

Output (data/training_8class_owncam/, the layout ml/train_yolo.py expects):
  train  base train + own train-session frames, repeated --repeat times so a
         few hundred own frames are not drowned out by thousands of base images
  valid  base valid + a held-out slice of the own train sessions
  test   ONLY the own test sessions (the real-camera score that matters);
         base test if no test session is given

Auto-capture saves a frame every few seconds, so neighbouring frames are near
copies. The own train sessions are split into bursts (frames less than --gap
seconds apart) and whole bursts go to train or valid, never both. Test sessions
must be separate capture sessions for the same reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.class_schema import NUM_TRAINING_CLASSES, training_names  # noqa: E402

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
FRAME_TIME = re.compile(r"(\d{8}_\d{6})_(\d{6})")
NAMES = training_names()


def frame_time(path: Path) -> float | None:
    match = FRAME_TIME.search(path.stem)
    if not match:
        return None
    stamp = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    return stamp.timestamp() + int(match.group(2)) / 1e6


def bursts(images: list[Path], gap: float) -> list[list[Path]]:
    """Consecutive frames < gap seconds apart form one burst; unnamed frames stand alone."""
    timed = sorted((t, p) for p in images if (t := frame_time(p)) is not None)
    groups, last = [], None
    for t, p in timed:
        if last is None or t - last >= gap:
            groups.append([])
        groups[-1].append(p)
        last = t
    groups += [[p] for p in images if frame_time(p) is None]
    return groups


def session_frames(session: Path) -> list[tuple[Path, Path]]:
    images_dir, labels_dir = session / "images", session / "labels"
    if not images_dir.is_dir():
        raise SystemExit(f"{session}: no images/ folder (run ml/prelabel_frames.py first)")
    pairs, missing, bad = [], [], []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            missing.append(image.name)
            continue
        for line in label.read_text().splitlines():
            parts = line.split()
            if parts and (len(parts) != 5 or not 0 <= int(float(parts[0])) < NUM_TRAINING_CLASSES):
                bad.append(f"{label.name}: {line!r}")
        pairs.append((image, label))
    if missing:
        raise SystemExit(f"{session}: {len(missing)} images have no label file, e.g. {missing[:3]}")
    if bad:
        raise SystemExit(f"{session}: labels outside the 8-class schema, e.g. {bad[:3]}")
    return pairs


def unreviewed(session: Path, pairs: list[tuple[Path, Path]]) -> int:
    """Label files still byte-identical to the model's draft from prelabel_frames.py."""
    drafts_file = session / "drafts.json"
    if not drafts_file.is_file():
        return 0
    drafts = json.loads(drafts_file.read_text())
    return sum(1 for _, label in pairs
               if drafts.get(label.stem) == hashlib.sha1(label.read_bytes()).hexdigest())


def copy_pair(image: Path, label: Path, out: Path, split: str, name: str) -> Counter:
    shutil.copy2(image, out / split / "images" / f"{name}{image.suffix.lower()}")
    shutil.copy2(label, out / split / "labels" / f"{name}.txt")
    return Counter(int(float(line.split()[0])) for line in label.read_text().splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="data/training_8class_multiscene")
    parser.add_argument("--train-sessions", nargs="*", default=[])
    parser.add_argument("--test-sessions", nargs="*", default=[])
    parser.add_argument("--out", default="data/training_8class_owncam")
    parser.add_argument("--valid-frac", type=float, default=0.15)
    parser.add_argument("--repeat", type=int, default=3, help="copies of each own train frame")
    parser.add_argument("--gap", type=float, default=10.0, help="seconds between frames that starts a new burst")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base, out = Path(args.base).resolve(), Path(args.out).resolve()
    train_sessions = [Path(s).resolve() for s in args.train_sessions]
    test_sessions = [Path(s).resolve() for s in args.test_sessions]
    if set(train_sessions) & set(test_sessions):
        raise SystemExit("a session cannot be both train and test")
    if out == base or out in train_sessions + test_sessions:
        raise SystemExit("--out must be a new folder")
    for split in SPLITS:
        if not (base / split / "images").is_dir():
            raise SystemExit(f"base dataset missing {base / split / 'images'}")

    sessions = {s: session_frames(s) for s in train_sessions + test_sessions}
    for s, pairs in sessions.items():
        pending = unreviewed(s, pairs)
        role = "TEST" if s in test_sessions else "train"
        print(f"{role:5} session {s.name}: {len(pairs)} frames"
              + (f"  WARNING {pending} label files are the unedited model draft" if pending else ""))

    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        for leaf in ("images", "labels"):
            (out / split / leaf).mkdir(parents=True)

    counts = {split: Counter() for split in SPLITS}
    images = {split: Counter() for split in SPLITS}
    base_splits = ("train", "valid") if test_sessions else SPLITS
    for split in base_splits:
        for image in sorted((base / split / "images").iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = base / split / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                label = out / ".empty.txt"
                label.touch()
            counts[split] += copy_pair(image, label, out, split, f"base__{image.stem}")
            images[split]["base"] += 1
    (out / ".empty.txt").unlink(missing_ok=True)

    rng = random.Random(args.seed)
    for s in train_sessions:
        labels = dict(sessions[s])
        groups = bursts(list(labels), args.gap)
        rng.shuffle(groups)
        n_valid = round(len(groups) * args.valid_frac) if len(groups) > 1 else 0
        for i, group in enumerate(groups):
            split = "valid" if i < n_valid else "train"
            for image in group:
                for k in range(args.repeat if split == "train" else 1):
                    counts[split] += copy_pair(image, labels[image], out, split, f"own_{s.name}__{image.stem}__r{k}")
                images[split]["own"] += 1
    for s in test_sessions:
        for image, label in sessions[s]:
            counts["test"] += copy_pair(image, label, out, "test", f"own_{s.name}__{image.stem}")
            images["test"]["own"] += 1

    print(f"\n{out}")
    print(f"  {'split':<7}{'base':>7}{'own':>6}" + "".join(f"{NAMES[i][:10]:>11}" for i in range(NUM_TRAINING_CLASSES)))
    for split in SPLITS:
        print(f"  {split:<7}{images[split]['base']:>7}{images[split]['own']:>6}"
              + "".join(f"{counts[split][i]:>11}" for i in range(NUM_TRAINING_CLASSES)))
    print(f"  own train frames are counted once above but copied {args.repeat}x (box counts include the copies)")
    print("  test = " + ("own camera only" if test_sessions else "base test (no --test-sessions given)"))
    print(f"\nnext: python ml/train_yolo.py --dataset-root {out} --model <weights> --epochs 60 --name <run>")


if __name__ == "__main__":
    main()
